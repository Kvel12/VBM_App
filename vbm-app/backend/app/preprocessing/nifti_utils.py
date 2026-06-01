"""
nifti_utils.py — Validación y utilidades para archivos NIfTI
Verifica que la imagen T1 sea válida antes de entrar al pipeline.
"""

import gzip
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

from app.config import TMP_DIR, VBM_PARAMS


# ─── Validación y carga ───────────────────────────────────────────────────────

def validate_and_load(nii_path: Path) -> Path:
    """
    Valida que el archivo sea un NIfTI T1 utilizable y lo descomprime
    si viene en .nii.gz. Retorna la ruta al .nii descomprimido.

    Checks:
    - El archivo existe y no está vacío
    - Es un NIfTI válido (nibabel puede leerlo)
    - Es 3D (no 4D funcional)
    - Tiene dimensiones razonables para una T1 estructural
    - Los valores de intensidad no son todos cero

    Raises:
        ValueError: si alguna validación falla, con mensaje descriptivo.
    """
    nii_path = Path(nii_path)

    if not nii_path.exists():
        raise ValueError(f"Archivo no encontrado: {nii_path.name}")
    if nii_path.stat().st_size == 0:
        raise ValueError(f"El archivo está vacío: {nii_path.name}")

    # Descomprimir .nii.gz si es necesario
    if nii_path.suffix == ".gz":
        nii_path = _decompress_gz(nii_path)

    # Intentar cargar con nibabel
    try:
        img = nib.load(str(nii_path))
    except Exception as e:
        raise ValueError(f"No es un archivo NIfTI válido: {e}")

    shape = img.shape

    # Debe ser 3D
    if len(shape) < 3:
        raise ValueError(
            f"La imagen tiene {len(shape)} dimensiones. Se esperan 3 (T1 estructural)."
        )
    if len(shape) == 4 and shape[3] > 1:
        raise ValueError(
            f"La imagen es 4D ({shape}). Solo se aceptan imágenes T1 estructurales 3D."
        )

    # Dimensiones mínimas razonables para una T1 cerebral
    min_dim = 64
    if any(d < min_dim for d in shape[:3]):
        raise ValueError(
            f"Dimensiones demasiado pequeñas: {shape[:3]}. "
            f"Se esperan al menos {min_dim} vóxeles por eje."
        )

    # Dimensiones máximas (evitar imágenes de cuerpo entero o mal formateo)
    max_dim = 600
    if any(d > max_dim for d in shape[:3]):
        raise ValueError(
            f"Dimensiones inusualmente grandes: {shape[:3]}. "
            f"Verifica que sea una imagen T1 cerebral."
        )

    # Verificar que no sea todo ceros
    try:
        data = np.asarray(img.dataobj, dtype=np.float32)
        if np.max(np.abs(data)) < 1e-6:
            raise ValueError("La imagen no contiene datos de intensidad (todos los vóxeles son cero).")
    except MemoryError:
        # Imagen muy grande — no podemos validar los datos, continuamos
        pass

    return nii_path


def _decompress_gz(gz_path: Path) -> Path:
    """
    Descomprime un .nii.gz al mismo directorio y retorna la ruta del .nii.
    Si el .nii ya existe (de una ejecución previa), lo reutiliza.
    """
    # Manejar tanto sub.nii.gz como sub.nii (doble extensión)
    if gz_path.stem.endswith(".nii"):
        out_path = gz_path.parent / gz_path.stem   # quita .gz → .nii
    else:
        out_path = gz_path.parent / (gz_path.stem + ".nii")

    if out_path.exists():
        return out_path

    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    return out_path


# ─── Información de la imagen ─────────────────────────────────────────────────

def get_image_info(nii_path: Path) -> dict:
    """
    Retorna un dict con información básica de la imagen NIfTI.
    Útil para logging y diagnóstico.
    """
    img    = nib.load(str(nii_path))
    vox    = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    return {
        "shape":      img.shape,
        "voxel_size": tuple(round(float(v), 3) for v in vox),
        "dtype":      str(img.get_data_dtype()),
        "affine_ok":  not np.allclose(img.affine, np.eye(4)),
    }


# ─── Features volumétricos desde mwp1 ────────────────────────────────────────

def extract_volumetric_features(gm_map_path: Path) -> dict:
    """
    Extrae features escalares del mapa GM modulado (mwp1) producido por SPM12.
    Estos features alimentan el SVM en el pipeline híbrido.

    El umbral 0.1 es el mismo usado en el script MATLAB de entrenamiento
    (run_vbm_spm12_checkpoint.m, Fase 4).

    Returns:
        dict con gm_volume_mm3, gm_volume_cm3, gm_mean_density,
        gm_std_density, gm_voxels, percentiles y shape.
    """
    threshold = VBM_PARAMS["gm_threshold"]  # 0.1

    img  = nib.load(str(gm_map_path))
    data = np.asarray(img.dataobj, dtype=np.float32)

    # Tamaño de vóxel en mm
    vox     = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))
    vox_vol = float(np.prod(vox))

    # Máscara GM
    mask    = data > threshold
    n_vox   = int(np.sum(mask))

    if n_vox == 0:
        raise ValueError(
            f"No se encontraron vóxeles GM por encima del umbral {threshold}. "
            "Verifica que el mapa GM sea válido."
        )

    gm_vals   = data[mask]
    gm_vol    = n_vox * vox_vol

    features = {
        "gm_volume_mm3":    round(gm_vol, 2),
        "gm_volume_cm3":    round(gm_vol / 1000, 4),
        "gm_mean_density":  float(np.mean(gm_vals)),
        "gm_std_density":   float(np.std(gm_vals)),
        "gm_voxels":        n_vox,
        "gm_p10":           float(np.percentile(gm_vals, 10)),
        "gm_p25":           float(np.percentile(gm_vals, 25)),
        "gm_p50":           float(np.percentile(gm_vals, 50)),
        "gm_p75":           float(np.percentile(gm_vals, 75)),
        "gm_p90":           float(np.percentile(gm_vals, 90)),
        "gm_max":           float(np.max(gm_vals)),
        "shape":            data.shape,
        "voxel_size_mm":    tuple(round(float(v), 3) for v in vox),
    }

    # Verificar shape esperado
    expected = tuple(VBM_PARAMS["expected_shape"])
    if data.shape != expected:
        # No es error fatal — solo advertencia, puede haber diferencia por bb
        features["shape_warning"] = (
            f"Shape {data.shape} difiere del esperado {expected}. "
            "Verifica los parámetros de normalización."
        )

    return features