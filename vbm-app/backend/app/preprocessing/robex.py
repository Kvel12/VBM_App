"""
robex.py — Skull stripping con ROBEX (paso OPCIONAL)

DECISIÓN DE DISEÑO CRÍTICA:
  El pipeline de entrenamiento (run_vbm_spm12_checkpoint.m en el iMac) pasaba
  la T1 DIRECTAMENTE a SPM12 Unified Segmentation sin skull stripping previo.
  SPM12 hace su propia extracción de cerebro internamente.

  Agregar ROBEX antes de SPM12 en inferencia produce mapas GM diferentes a los
  del entrenamiento (~28.000 vóxeles extra para sub-102), lo que degrada las
  predicciones del CNN.

  POR TANTO: skull_strip() retorna la T1 cruda por defecto (skip_robex=True).
  ROBEX solo se usaría si en el futuro se reentrenara el modelo con ROBEX
  como paso previo explícito.

  La función se mantiene implementada para no romper la arquitectura del
  backend ni el frontend (que muestra "Extracción de cráneo · ROBEX" como paso).
  En la UI ese paso se completa instantáneamente con el mensaje correcto.
"""

import time
from pathlib import Path

from app.config import ROBEX_DIR, ROBEX_SCRIPT, TMP_DIR


def skull_strip(t1_path: Path, job_id: str,
                skip_robex: bool = True) -> Path:
    """
    Paso de skull stripping del pipeline.

    Por defecto skip_robex=True: retorna la T1 original sin modificar.
    Esto replica exactamente el pipeline de entrenamiento (SPM12 directo).

    Args:
        t1_path:    Ruta al .nii de entrada (ya descomprimido).
        job_id:     ID del job (para logging).
        skip_robex: Si True (defecto), retorna t1_path sin modificar.
                    SPM12 Unified Segmentation hace su propia extracción
                    de cerebro internamente — no se necesita ROBEX previo.

    Returns:
        Path a la imagen que entrará a SPM12 (cruda o stripped).
    """
    t1_path = Path(t1_path)

    if skip_robex:
        print(f"[ROBEX] Paso omitido — T1 pasa directamente a SPM12 "
              f"(replica pipeline de entrenamiento)")
        return t1_path

    # ── Skull stripping activo (solo si skip_robex=False) ─────────────────────
    import subprocess

    job_dir       = TMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    stripped_path = job_dir / f"{job_id}_brain.nii"
    mask_path     = job_dir / f"{job_id}_brain_mask.nii"

    if stripped_path.exists():
        return stripped_path

    if not ROBEX_SCRIPT.exists():
        print(f"[ROBEX] Binario no encontrado en {ROBEX_SCRIPT} — usando T1 cruda")
        return t1_path

    cmd = [
        str(ROBEX_SCRIPT),
        str(t1_path),
        str(stripped_path),
        str(mask_path),
    ]

    print(f"[ROBEX] Ejecutando skull stripping: {t1_path.name}")
    t_start = time.time()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(ROBEX_DIR),
        )
        elapsed = round(time.time() - t_start, 1)

        if result.returncode != 0 or not stripped_path.exists():
            print(f"[ROBEX] Falló (código {result.returncode}) — usando T1 cruda")
            return t1_path

        print(f"[ROBEX] Completado en {elapsed}s")
        return stripped_path

    except Exception as e:
        print(f"[ROBEX] Error: {e} — usando T1 cruda")
        return t1_path
