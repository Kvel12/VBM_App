"""
robex.py — Wrapper de skull stripping con ROBEX
Llama al binario ROBEX instalado en /opt/ROBEX dentro del contenedor.

ROBEX produce dos archivos:
  - <output>.nii   : imagen cerebro extraído (sin cráneo)
  - <output>_mask.nii : máscara binaria del cerebro

El pipeline SPM12 recibe la imagen extraída como entrada para segmentación.

NOTA SOBRE ROBEX EN ESTE PIPELINE:
  En la tesis se descartó el dataset IXI-HH precisamente porque ROBEX
  producía over-stripping que reducía artificialmente el volumen de GM.
  Sin embargo, el pipeline de entrenamiento SPM12 en el iMac NO usó ROBEX
  como paso previo — SPM12 Unified Segmentation recibe la T1 cruda y hace
  su propia segmentación interna.

  ROBEX se incluye aquí como paso opcional de QC porque en inferencia
  la T1 puede venir de cualquier escáner. Si ROBEX falla o produce
  resultados anómalos, el pipeline puede continuar con la T1 cruda
  (fallback_to_raw=True).
"""

import subprocess
import shutil
import time
from pathlib import Path

from app.config import ROBEX_DIR, ROBEX_SCRIPT, TMP_DIR


# ─── Parámetros de QC post-ROBEX ──────────────────────────────────────────────
# Si el volumen del cerebro extraído es menor que este % del volumen total,
# ROBEX probablemente hizo over-stripping → usar T1 cruda como fallback
MIN_BRAIN_RATIO  = 0.10   # mínimo 10% del volumen total debe ser cerebro
MAX_BRAIN_RATIO  = 0.60   # máximo 60% (más indica que no se eliminó el cráneo)
ROBEX_TIMEOUT_S  = 300    # 5 minutos máximo para skull stripping


def skull_strip(t1_path: Path, job_id: str,
                fallback_to_raw: bool = True) -> Path:
    """
    Ejecuta ROBEX sobre la imagen T1.

    Args:
        t1_path:        Ruta al .nii de entrada (ya descomprimido).
        job_id:         ID del job para naming de archivos temporales.
        fallback_to_raw: Si True y ROBEX falla, retorna t1_path sin modificar.
                         Si False, lanza excepción ante cualquier fallo.

    Returns:
        Path al .nii con cráneo eliminado (o t1_path si fallback).
    """
    t1_path = Path(t1_path)
    job_dir = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Paths de salida
    stripped_path = job_dir / f"{job_id}_brain.nii"
    mask_path     = job_dir / f"{job_id}_brain_mask.nii"

    # Si ya se procesó (reanudación de job), reusar
    if stripped_path.exists():
        return stripped_path

    # Verificar que el binario existe
    if not ROBEX_SCRIPT.exists():
        msg = f"ROBEX no encontrado en {ROBEX_SCRIPT}"
        if fallback_to_raw:
            print(f"[ROBEX] ADVERTENCIA: {msg}. Usando T1 cruda.")
            return t1_path
        raise RuntimeError(msg)

    # ── Ejecutar ROBEX ────────────────────────────────────────────────────────
    cmd = [
        str(ROBEX_SCRIPT),
        str(t1_path),        # input
        str(stripped_path),  # output brain
        str(mask_path),      # output mask
    ]

    print(f"[ROBEX] Iniciando skull stripping: {t1_path.name}")
    t_start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=ROBEX_TIMEOUT_S,
            cwd=str(ROBEX_DIR),   # ROBEX necesita correr desde su propio directorio
        )

        elapsed = round(time.time() - t_start, 1)

        if result.returncode != 0:
            raise RuntimeError(
                f"ROBEX salió con código {result.returncode}.\n"
                f"stderr: {result.stderr[:500]}"
            )

        if not stripped_path.exists():
            raise RuntimeError(
                f"ROBEX terminó sin error pero no generó el archivo de salida: {stripped_path}"
            )

        print(f"[ROBEX] Completado en {elapsed}s")

        # ── QC post-ROBEX ─────────────────────────────────────────────────────
        qc_ok, qc_msg = _qc_stripped(t1_path, stripped_path)
        if not qc_ok:
            print(f"[ROBEX] QC falló: {qc_msg}")
            if fallback_to_raw:
                print("[ROBEX] Usando T1 cruda como fallback.")
                return t1_path
            raise ValueError(f"QC de skull stripping falló: {qc_msg}")

        return stripped_path

    except subprocess.TimeoutExpired:
        msg = f"ROBEX excedió el timeout de {ROBEX_TIMEOUT_S}s"
        print(f"[ROBEX] ERROR: {msg}")
        if fallback_to_raw:
            print("[ROBEX] Usando T1 cruda como fallback.")
            return t1_path
        raise RuntimeError(msg)

    except Exception as e:
        print(f"[ROBEX] ERROR: {e}")
        if fallback_to_raw:
            print("[ROBEX] Usando T1 cruda como fallback.")
            return t1_path
        raise


# ─── QC post-extracción ───────────────────────────────────────────────────────

def _qc_stripped(original_path: Path, stripped_path: Path) -> tuple[bool, str]:
    """
    Verifica que ROBEX no haya hecho over-stripping ni under-stripping.

    Returns:
        (True, "") si el QC pasa.
        (False, mensaje) si hay problema.
    """
    try:
        import nibabel as nib
        import numpy as np

        orig     = nib.load(str(original_path))
        stripped = nib.load(str(stripped_path))

        orig_data     = np.asarray(orig.dataobj, dtype=np.float32)
        stripped_data = np.asarray(stripped.dataobj, dtype=np.float32)

        # Vóxeles no-cero como proxy de "tejido"
        total_vox  = np.sum(orig_data > np.percentile(orig_data, 20))
        brain_vox  = np.sum(stripped_data > 0)

        if total_vox == 0:
            return False, "La imagen original parece estar vacía."

        ratio = brain_vox / total_vox

        if ratio < MIN_BRAIN_RATIO:
            return False, (
                f"Over-stripping detectado: solo {ratio*100:.1f}% del volumen "
                f"quedó después de la extracción (mínimo esperado: {MIN_BRAIN_RATIO*100:.0f}%)."
            )

        if ratio > MAX_BRAIN_RATIO:
            return False, (
                f"Under-stripping detectado: {ratio*100:.1f}% del volumen original "
                f"quedó (máximo esperado: {MAX_BRAIN_RATIO*100:.0f}%). "
                "Es posible que el cráneo no se haya eliminado correctamente."
            )

        return True, ""

    except Exception as e:
        # Si el QC falla por error interno, no bloqueamos el pipeline
        print(f"[ROBEX QC] Error en verificación: {e}")
        return True, ""