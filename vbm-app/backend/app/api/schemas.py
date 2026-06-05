"""
schemas.py — Modelos Pydantic para request/response de la API
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal
from enum import Enum


class ModelType(str, Enum):
    DEEPMRIPREP = "deepmriprep"   # antes SPM12_DARTEL (descartado por irreproducibilidad cross-platform)
    NNUNET      = "nnunet"        # segmentación 3D nnU-Net (transfer learning desde modelo TBI)
    # HYBRID se descartó — features volumétricos globales no aportaron señal
    # discriminativa sobre los mapas de deepmriprep (p=0.24 vs p<0.001 con SPM12).
    # La arquitectura w_CNN * P(CNN) + w_SVM * P(SVM) determinó w_CNN=1.00 → fusión
    # se reduce al CNN solo. Documentado en la tesis como hallazgo experimental.


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

    # ── Predicción (CLASIFICACIÓN: deepmriprep) ──────────────────────────────
    # Opcionales porque el modelo de segmentación (nnUNet) no produce estos.
    prediction:        Optional[Literal["epilepsy", "control"]] = None
    confidence:        Optional[float] = Field(None, ge=0.0, le=1.0)
    prob_epilepsy:     Optional[float] = Field(None, ge=0.0, le=1.0)
    prob_control:      Optional[float] = Field(None, ge=0.0, le=1.0)

    # ── Métricas reportadas del modelo (clasificación) ───────────────────────
    model_auc:         Optional[float] = None
    model_sensitivity: Optional[float] = None
    model_specificity: Optional[float] = None
    model_accuracy:    Optional[float] = None

    # ── Features volumétricos del sujeto (clasificación) ─────────────────────
    gm_volume_cm3:     Optional[float] = None
    gm_mean_density:   Optional[float] = None
    gm_voxels:         Optional[int]   = None

    # ── Salida de SEGMENTACIÓN (nnUNet) ──────────────────────────────────────
    # mask_filename: nombre del .nii.gz dentro de tmp/<job_id>/nnunet_out/
    #                el frontend lo carga via GET /mask/{job_id}
    # t1_filename:   nombre del T1 que se le pasó a nnUNet (para el visor)
    mask_filename:     Optional[str]   = None
    t1_filename:       Optional[str]   = None
    mask_voxels:       Optional[int]   = None
    mask_volume_cm3:   Optional[float] = None
    n_clusters:        Optional[int]   = None   # clusters conexos en la máscara
    largest_cluster_cm3: Optional[float] = None

    # ── Métricas del modelo de segmentación (constantes, evaluación 778 sujetos) ─
    # DSC y HD95 son MEDIANAS; sensibilidad/especificidad a nivel sujeto.
    model_dsc:                Optional[float] = None
    model_hd95:               Optional[float] = None
    model_seg_sensitivity:    Optional[float] = None   # tasa de detección sujeto
    model_seg_specificity:    Optional[float] = None   # controles sin FP
    model_seg_ppv:            Optional[float] = None   # valor predictivo positivo
    model_seg_npv:            Optional[float] = None   # valor predictivo negativo

    # ── Metadatos del análisis ───────────────────────────────────────────────
    model_used:        ModelType
    test_name:         str
    patient_name:      Optional[str] = None
    notes:             Optional[str] = None
    processing_time_s: Optional[float] = None


# Necesario para la referencia forward en JobStatusResponse
JobStatusResponse.model_rebuild()