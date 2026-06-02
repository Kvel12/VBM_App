"""
schemas.py — Modelos Pydantic para request/response de la API
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal
from enum import Enum


class ModelType(str, Enum):
    DEEPMRIPREP = "deepmriprep"   # antes SPM12_DARTEL (descartado por irreproducibilidad cross-platform)
    HYBRID      = "hybrid"
    NNUNET      = "nnunet"


# ─── Request (multipart, el .nii/.gz va como UploadFile en la ruta) ───────────
class AnalysisMetadata(BaseModel):
    """Metadatos opcionales que el usuario ingresa en el formulario."""
    test_name:    str            = Field(..., description="Nombre de la prueba")
    patient_name: Optional[str] = Field(None, description="Nombre del paciente")
    notes:        Optional[str] = Field(None, description="Notas clínicas")
    model:        ModelType     = Field(..., description="Pipeline a ejecutar")
    use_robex:    bool          = Field(False, description=(
        "Aplicar ROBEX skull stripping antes de SPM12. Por defecto False — "
        "el pipeline de entrenamiento pasa la T1 cruda directamente a SPM12. "
        "Activar solo si la T1 viene ya con cráneo eliminado o por elección "
        "metodológica del usuario."
    ))


# ─── Respuesta de creación de job ─────────────────────────────────────────────
class JobCreatedResponse(BaseModel):
    job_id:  str
    message: str = "Job creado exitosamente"


# ─── Estado del job ───────────────────────────────────────────────────────────
class StepStatus(str, Enum):
    PENDING    = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED  = "completed"
    ERROR      = "error"


class ProcessStep(BaseModel):
    step:    int
    name:    str
    status:  StepStatus
    message: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id:   str
    status:   Literal["pending", "running", "completed", "error"]
    progress: int = Field(ge=0, le=100, description="Porcentaje 0-100")
    steps:    list[ProcessStep]
    result:   Optional["AnalysisResult"] = None
    error:    Optional[str] = None


# ─── Resultado final ──────────────────────────────────────────────────────────
class AnalysisResult(BaseModel):
    # Los campos model_* colisionan con el namespace protegido de Pydantic v2
    # ("model_*" suele ser reservado). Sin esto cada arranque imprime warnings.
    model_config = ConfigDict(protected_namespaces=())

    # Predicción
    prediction:        Literal["epilepsy", "control"]
    confidence:        float = Field(ge=0.0, le=1.0)
    prob_epilepsy:     float = Field(ge=0.0, le=1.0)
    prob_control:      float = Field(ge=0.0, le=1.0)

    # Métricas del modelo (del conjunto de validación, constantes)
    model_auc:         float
    model_sensitivity: float
    model_specificity: float
    model_accuracy:    float

    # Features volumétricos extraídos de la imagen
    gm_volume_cm3:     Optional[float] = None
    gm_mean_density:   Optional[float] = None
    gm_voxels:         Optional[int]   = None

    # Metadatos del análisis
    model_used:        ModelType
    test_name:         str
    patient_name:      Optional[str] = None
    notes:             Optional[str] = None
    processing_time_s: Optional[float] = None


# Necesario para la referencia forward en JobStatusResponse
JobStatusResponse.model_rebuild()