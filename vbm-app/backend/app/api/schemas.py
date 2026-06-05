"""
schemas.py — Pydantic models for API request/response
"""

from pydantic import BaseModel, ConfigDict, Field
from typing import Optional, Literal
from enum import Enum


class ModelType(str, Enum):
    DEEPMRIPREP = "deepmriprep"   # previously SPM12_DARTEL (discarded due to cross-platform irreproducibility)
    NNUNET      = "nnunet"        # 3D nnU-Net segmentation (transfer learning from the TBI model)
    # HYBRID was discarded — global volumetric features added no discriminative
    # signal on top of deepmriprep maps (p=0.24 vs p<0.001 with SPM12).
    # The architecture w_CNN * P(CNN) + w_SVM * P(SVM) yielded w_CNN=1.00 →
    # fusion collapses to the CNN alone. Documented in the thesis as an
    # experimental finding.


# ─── Request (multipart, the .nii/.gz arrives as UploadFile in the route) ────
class AnalysisMetadata(BaseModel):
    """Optional metadata the user enters in the form."""
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


# ─── Job creation response ───────────────────────────────────────────────────
class JobCreatedResponse(BaseModel):
    job_id:  str
    message: str = "Job creado exitosamente"


# ─── Job status ──────────────────────────────────────────────────────────────
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


# ─── Final result ────────────────────────────────────────────────────────────
class AnalysisResult(BaseModel):
    # The model_* fields collide with Pydantic v2's protected namespace
    # ("model_*" is usually reserved). Without this, every startup prints warnings.
    model_config = ConfigDict(protected_namespaces=())

    # ── Prediction (CLASSIFICATION: deepmriprep) ─────────────────────────────
    # Optional because the segmentation model (nnUNet) does not produce these.
    prediction:        Optional[Literal["epilepsy", "control"]] = None
    confidence:        Optional[float] = Field(None, ge=0.0, le=1.0)
    prob_epilepsy:     Optional[float] = Field(None, ge=0.0, le=1.0)
    prob_control:      Optional[float] = Field(None, ge=0.0, le=1.0)

    # ── Reported model metrics (classification) ──────────────────────────────
    model_auc:         Optional[float] = None
    model_sensitivity: Optional[float] = None
    model_specificity: Optional[float] = None
    model_accuracy:    Optional[float] = None

    # ── Subject volumetric features (classification) ─────────────────────────
    gm_volume_cm3:     Optional[float] = None
    gm_mean_density:   Optional[float] = None
    gm_voxels:         Optional[int]   = None

    # ── SEGMENTATION output (nnUNet) ─────────────────────────────────────────
    # mask_filename: name of the .nii.gz inside tmp/<job_id>/nnunet_out/
    #                the frontend loads it via GET /mask/{job_id}
    # t1_filename:   name of the T1 passed to nnUNet (for the viewer)
    mask_filename:     Optional[str]   = None
    t1_filename:       Optional[str]   = None
    mask_voxels:       Optional[int]   = None
    mask_volume_cm3:   Optional[float] = None
    n_clusters:        Optional[int]   = None   # connected clusters in the mask
    largest_cluster_cm3: Optional[float] = None

    # ── Segmentation model metrics (constants, evaluation on 778 subjects) ───
    # DSC and HD95 are MEDIANS; sensitivity/specificity at subject level.
    model_dsc:                Optional[float] = None
    model_hd95:               Optional[float] = None
    model_seg_sensitivity:    Optional[float] = None   # subject-level detection rate
    model_seg_specificity:    Optional[float] = None   # controls without FP
    model_seg_ppv:            Optional[float] = None   # positive predictive value
    model_seg_npv:            Optional[float] = None   # negative predictive value

    # ── Analysis metadata ────────────────────────────────────────────────────
    model_used:        ModelType
    test_name:         str
    patient_name:      Optional[str] = None
    notes:             Optional[str] = None
    processing_time_s: Optional[float] = None


# Required for the forward reference in JobStatusResponse
JobStatusResponse.model_rebuild()
