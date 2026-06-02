"""
nnunet.py — Pipeline nnUNet (segmentación de zonas epileptogénicas)

ESTADO ACTUAL: Placeholder estructurado.
nnUNet no fue implementado en el trabajo de tesis actual — está marcado
como extensión futura (3D-nnUNet de Kersting et al. 2024/2025).
Este módulo tiene la estructura completa lista para implementar
cuando los pesos estén disponibles.

El endpoint de la API ya funciona y retorna un error descriptivo
si se intenta usar este modelo antes de implementarlo.
"""

from pathlib import Path
from typing import Callable

from app.api.schemas import AnalysisResult, ModelType, StepStatus


def run(brain_path: Path, job_id: str,
        update_step: Callable, metadata: dict) -> AnalysisResult:
    """
    Pipeline nnUNet — pendiente de implementación.

    Cuando se implemente, el flujo será:
      1. Cargando imagen T1         (routes.py)
      2. Extracción de cráneo ROBEX (routes.py)
      3. Segmentación nnUNet        ← aquí: inferencia con nnunet predict
      4. Análisis de máscara        ← aquí: volumetría de la máscara generada

    Para implementar:
      1. Instalar nnunetv2 en requirements.txt
      2. Copiar pesos a backend/models/nnunet_weights/
      3. Llamar: nnUNetv2_predict -i <input> -o <output> -d <dataset_id>
         -c 3d_fullres -f all --save_probabilities
      4. Analizar la máscara de salida para extraer volumen de zona epileptogénica
    """
    update_step(job_id, 3, StepStatus.ERROR,
                "nnUNet no implementado en esta versión")

    raise NotImplementedError(
        "El modelo nnUNet está marcado como extensión futura y no está "
        "disponible en esta versión de la aplicación. "
        "Usa SPM12/DARTEL o el Modelo Híbrido."
    )