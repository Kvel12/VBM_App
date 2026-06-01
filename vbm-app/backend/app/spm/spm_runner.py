"""
spm_runner.py — Ejecutor de SPM12 Standalone desde Python

Invoca el binario run_spm12.sh con un archivo de batch .m generado
dinámicamente para cada sujeto nuevo.

Flujo para un sujeto nuevo:
  1. run_segmentation(t1_path, job_id)
     → genera rc1*.nii y rc2*.nii en job_dir
  2. run_dartel_normalise(rc1_path, ff_path, job_id)
     → genera mwp1*.nii (mapa GM modulado en espacio MNI)

El Template DARTEL ya está congelado (entrenado con 778 sujetos).
Aquí solo se usa "Run DARTEL (existing templates)" — nunca se recrea el template.
"""

import subprocess
import time
import shutil
import tempfile
from pathlib import Path
from string import Template

from app.config import (
    SPM_RUN_SCRIPT, MCR_DIR, SPM_STANDALONE_DIR,
    DARTEL_TEMPLATES, DARTEL_TEMPLATE_6,
    VBM_PARAMS, SEG_PARAMS, TMP_DIR,
)

SPM_TIMEOUT_SEG    = 600   # 10 min para segmentación
SPM_TIMEOUT_NORM   = 300   # 5 min para normalización


# ─── Batch templates (se renderizan con los paths reales en runtime) ──────────

_BATCH_SEGMENT = Template("""
%% SPM12 Unified Segmentation — generado automáticamente
%% Job: $job_id

addpath('$spm_dir');
spm('defaults', 'FMRI');
spm_jobman('initcfg');

TPM_PATH = fullfile(spm('Dir'), 'tpm', 'TPM.nii');

matlabbatch{1}.spm.spatial.preproc.channel.vols     = {'$t1_path'};
matlabbatch{1}.spm.spatial.preproc.channel.biasreg  = $biasreg;
matlabbatch{1}.spm.spatial.preproc.channel.biasfwhm = $biasfwhm;
matlabbatch{1}.spm.spatial.preproc.channel.write    = [0 0];

ngaus_vals = [$ngaus];
for c = 1:6
    matlabbatch{1}.spm.spatial.preproc.tissue(c).tpm   = {[TPM_PATH ',' num2str(c)]};
    matlabbatch{1}.spm.spatial.preproc.tissue(c).ngaus = ngaus_vals(c);
    if c <= 2
        matlabbatch{1}.spm.spatial.preproc.tissue(c).native = [1 1];
    else
        matlabbatch{1}.spm.spatial.preproc.tissue(c).native = [0 0];
    end
    matlabbatch{1}.spm.spatial.preproc.tissue(c).warped = [0 0];
end

matlabbatch{1}.spm.spatial.preproc.warp.mrf     = $mrf;
matlabbatch{1}.spm.spatial.preproc.warp.cleanup = $cleanup;
matlabbatch{1}.spm.spatial.preproc.warp.reg     = [$reg];
matlabbatch{1}.spm.spatial.preproc.warp.affreg  = '$affreg';
matlabbatch{1}.spm.spatial.preproc.warp.fwhm    = $fwhm_seg;
matlabbatch{1}.spm.spatial.preproc.warp.samp    = $samp;
matlabbatch{1}.spm.spatial.preproc.warp.write   = [0 0];

spm_jobman('run', matlabbatch);
exit;
""")

_BATCH_DARTEL_NORM = Template("""
%% SPM12 DARTEL Normalise to MNI — generado automáticamente
%% Job: $job_id

addpath('$spm_dir');
spm('defaults', 'FMRI');
spm_jobman('initcfg');

matlabbatch{1}.spm.tools.dartel.mni_norm.template = {'$template_6'};
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.flowfields = {'$ff_path'};
matlabbatch{1}.spm.tools.dartel.mni_norm.data.subjs.images     = {{'$rc1_path'}};
matlabbatch{1}.spm.tools.dartel.mni_norm.vox      = [$vox];
matlabbatch{1}.spm.tools.dartel.mni_norm.bb       = [$bb_row1; $bb_row2];
matlabbatch{1}.spm.tools.dartel.mni_norm.preserve = $preserve;
matlabbatch{1}.spm.tools.dartel.mni_norm.fwhm     = [$fwhm];

spm_jobman('run', matlabbatch);
exit;
""")


# ─── Runner interno ───────────────────────────────────────────────────────────

def _run_spm_batch(batch_content: str, job_id: str, step: str,
                   timeout: int) -> None:
    """
    Escribe el batch a un archivo temporal y lo ejecuta con SPM Standalone.

    SPM Standalone se invoca como:
        ./run_spm12.sh <mcr_dir> batch <batch_file.m>
    """
    # Escribir batch a archivo temporal en el directorio del job
    job_dir    = TMP_DIR / job_id
    batch_path = job_dir / f"batch_{step}.m"
    batch_path.write_text(batch_content, encoding="utf-8")

    cmd = [
        str(SPM_RUN_SCRIPT),
        str(MCR_DIR),
        "batch",
        str(batch_path),
    ]

    print(f"[SPM] Ejecutando paso '{step}' para job {job_id}")
    print(f"[SPM] Comando: {' '.join(cmd)}")
    t_start = time.time()

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(SPM_STANDALONE_DIR),
    )

    elapsed = round(time.time() - t_start, 1)
    print(f"[SPM] '{step}' completado en {elapsed}s (returncode={result.returncode})")

    if result.returncode != 0:
        # Incluir las últimas líneas de stderr para diagnóstico
        stderr_tail = result.stderr[-1000:] if result.stderr else "(sin stderr)"
        stdout_tail = result.stdout[-500:]  if result.stdout else "(sin stdout)"
        raise RuntimeError(
            f"SPM '{step}' falló (código {result.returncode}).\n"
            f"--- stderr (últimas líneas) ---\n{stderr_tail}\n"
            f"--- stdout (últimas líneas) ---\n{stdout_tail}"
        )


# ─── Segmentación ─────────────────────────────────────────────────────────────

def run_segmentation(t1_path: Path, job_id: str) -> tuple[Path, Path, Path]:
    """
    Ejecuta SPM12 Unified Segmentation sobre la T1.

    Produce en el mismo directorio que t1_path:
        rc1<nombre>.nii  — GM en espacio DARTEL (para normalización)
        rc2<nombre>.nii  — WM en espacio DARTEL
        c1<nombre>.nii   — GM en espacio nativo (opcional, para QC)

    Returns:
        (rc1_path, rc2_path, c1_path)
    """
    t1_path = Path(t1_path)
    seg_p   = SEG_PARAMS
    vbm_p   = VBM_PARAMS

    batch = _BATCH_SEGMENT.substitute(
        job_id    = job_id,
        spm_dir   = str(SPM_STANDALONE_DIR),
        t1_path   = str(t1_path),
        biasreg   = seg_p["biasreg"],
        biasfwhm  = seg_p["biasfwhm"],
        ngaus     = " ".join(str(n) for n in seg_p["ngaus"]),
        mrf       = seg_p["mrf"],
        cleanup   = seg_p["cleanup"],
        reg       = " ".join(str(r) for r in seg_p["reg"]),
        affreg    = seg_p["affreg"],
        fwhm_seg  = seg_p.get("fwhm", 0),
        samp      = seg_p["samp"],
    )

    _run_spm_batch(batch, job_id, "segmentation", SPM_TIMEOUT_SEG)

    # SPM genera los outputs en el mismo directorio que la T1
    stem   = t1_path.stem
    t1_dir = t1_path.parent

    rc1 = t1_dir / f"rc1{stem}.nii"
    rc2 = t1_dir / f"rc2{stem}.nii"
    c1  = t1_dir / f"c1{stem}.nii"

    if not rc1.exists():
        raise RuntimeError(
            f"SPM segmentación no generó rc1: {rc1}\n"
            "Verifica que la T1 sea válida y que el TPM esté accesible."
        )
    if not rc2.exists():
        raise RuntimeError(f"SPM segmentación no generó rc2: {rc2}")

    return rc1, rc2, c1


# ─── Normalización DARTEL a MNI ───────────────────────────────────────────────

def run_dartel_normalise(rc1_path: Path, ff_path: Path,
                         job_id: str) -> Path:
    """
    Ejecuta DARTEL Normalise to MNI Space con templates congelados.

    Args:
        rc1_path: GM en espacio DARTEL (rc1*.nii de la segmentación)
        ff_path:  Flow field del sujeto (u_rc1*.nii generado por DARTEL existing)
        job_id:   ID del job

    Returns:
        Path al mapa GM modulado en espacio MNI (mwrc1*.nii o mwc1*.nii)
    """
    rc1_path = Path(rc1_path)
    ff_path  = Path(ff_path)
    vbm_p    = VBM_PARAMS

    bb = vbm_p["bb"]
    bb_row1 = " ".join(str(v) for v in bb[0])
    bb_row2 = " ".join(str(v) for v in bb[1])

    batch = _BATCH_DARTEL_NORM.substitute(
        job_id     = job_id,
        spm_dir    = str(SPM_STANDALONE_DIR),
        template_6 = str(DARTEL_TEMPLATE_6),
        ff_path    = str(ff_path),
        rc1_path   = str(rc1_path),
        vox        = " ".join(str(v) for v in vbm_p["vox"]),
        bb_row1    = bb_row1,
        bb_row2    = bb_row2,
        preserve   = vbm_p["preserve"],
        fwhm       = " ".join(str(v) for v in vbm_p["fwhm"]),
    )

    _run_spm_batch(batch, job_id, "dartel_norm", SPM_TIMEOUT_NORM)

    # Buscar el output — SPM lo coloca junto a rc1
    stem    = rc1_path.stem   # rc1<nombre>
    out_dir = rc1_path.parent

    # Posibles nombres según versión de SPM
    candidates = [
        out_dir / f"mwrc1{stem[3:]}.nii",   # mwrc1<nombre>.nii
        out_dir / f"mwc1{stem[3:]}.nii",    # mwc1<nombre>.nii
        out_dir / f"mw{stem}.nii",          # mwrc1<stem completo>
    ]

    # Búsqueda por glob como fallback
    glob_results = list(out_dir.glob(f"mw*{stem[3:]}*.nii"))

    for candidate in candidates + glob_results:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        f"DARTEL normalización no generó el mapa GM esperado.\n"
        f"Buscado en: {out_dir}\n"
        f"Candidatos probados: {[str(c) for c in candidates]}\n"
        f"Archivos mw* encontrados: {[f.name for f in out_dir.glob('mw*.nii')]}"
    )


# ─── Pipeline DARTEL existing completo ───────────────────────────────────────
# Para un sujeto nuevo, DARTEL existing genera el flow field Y normaliza
# en un solo job. Esta función lo hace todo en un paso.

_BATCH_DARTEL_EXISTING = Template("""
%% SPM12 DARTEL existing templates — generado automáticamente
%% Job: $job_id

addpath('$spm_dir');
spm('defaults', 'FMRI');
spm_jobman('initcfg');

%% Paso 1: Run DARTEL (existing templates) → genera flow field u_rc1*.nii
matlabbatch{1}.spm.tools.dartel.warp1.images = {
    {'$rc1_path'},
    {'$rc2_path'}
};
matlabbatch{1}.spm.tools.dartel.warp1.settings.rform    = 0;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).its    = 3;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).rparam = [4 2 1e-6];
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).K      = 0;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(1).template = {'${template_1}'};
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).its    = 3;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).rparam = [2 1 1e-6];
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).K      = 0;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(2).template = {'${template_2}'};
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).its    = 3;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).rparam = [1 0.5 1e-6];
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).K      = 1;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(3).template = {'${template_3}'};
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).its    = 3;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).rparam = [0.5 0.25 1e-6];
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).K      = 2;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(4).template = {'${template_4}'};
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).its    = 3;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).rparam = [0.25 0.125 1e-6];
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).K      = 4;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(5).template = {'${template_5}'};
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).its    = 3;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).rparam = [0.25 0.125 1e-6];
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).K      = 6;
matlabbatch{1}.spm.tools.dartel.warp1.settings.param(6).template = {'${template_6}'};
matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.lmreg = 0.01;
matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.cyc   = 3;
matlabbatch{1}.spm.tools.dartel.warp1.settings.optim.its   = 3;

%% Paso 2: Normalise to MNI Space
matlabbatch{2}.spm.tools.dartel.mni_norm.template = {'$template_6'};
matlabbatch{2}.spm.tools.dartel.mni_norm.data.subjs.flowfields(1) = ...
    cfg_dep('Run DARTEL (existing templates): Flow Field', ...
            substruct('.','val', '{}',{1}, '.','val', '{}',{1}, ...
                      '.','val', '{}',{1}, '.','val', '{}',{1}), ...
            substruct('.','files', '{}',{1}));
matlabbatch{2}.spm.tools.dartel.mni_norm.data.subjs.images = {{'$rc1_path'}};
matlabbatch{2}.spm.tools.dartel.mni_norm.vox      = [$vox];
matlabbatch{2}.spm.tools.dartel.mni_norm.bb       = [$bb_row1; $bb_row2];
matlabbatch{2}.spm.tools.dartel.mni_norm.preserve = $preserve;
matlabbatch{2}.spm.tools.dartel.mni_norm.fwhm     = [$fwhm];

spm_jobman('run', matlabbatch);
exit;
""")


def run_dartel_existing_full(rc1_path: Path, rc2_path: Path,
                              job_id: str) -> tuple[Path, Path]:
    """
    Ejecuta Run DARTEL (existing templates) + Normalise to MNI en un solo job.
    Este es el flujo correcto para sujetos nuevos.

    Returns:
        (gm_map_path, ff_path) — mapa GM modulado y flow field generado
    """
    rc1_path = Path(rc1_path)
    rc2_path = Path(rc2_path)
    vbm_p    = VBM_PARAMS
    bb       = vbm_p["bb"]

    batch = _BATCH_DARTEL_EXISTING.substitute(
        job_id     = job_id,
        spm_dir    = str(SPM_STANDALONE_DIR),
        rc1_path   = str(rc1_path),
        rc2_path   = str(rc2_path),
        template_1 = str(DARTEL_TEMPLATES[1]),
        template_2 = str(DARTEL_TEMPLATES[2]),
        template_3 = str(DARTEL_TEMPLATES[3]),
        template_4 = str(DARTEL_TEMPLATES[4]),
        template_5 = str(DARTEL_TEMPLATES[5]),
        template_6 = str(DARTEL_TEMPLATES[6]),
        vox        = " ".join(str(v) for v in vbm_p["vox"]),
        bb_row1    = " ".join(str(v) for v in bb[0]),
        bb_row2    = " ".join(str(v) for v in bb[1]),
        preserve   = vbm_p["preserve"],
        fwhm       = " ".join(str(v) for v in vbm_p["fwhm"]),
    )

    # DARTEL existing puede tardar más que solo la normalización
    _run_spm_batch(batch, job_id, "dartel_existing_full",
                   timeout=SPM_TIMEOUT_SEG + SPM_TIMEOUT_NORM)

    # Buscar flow field generado (u_rc1*.nii)
    stem    = rc1_path.stem   # rc1<nombre>
    out_dir = rc1_path.parent

    ff_candidates = list(out_dir.glob(f"u_{stem}*.nii"))
    if not ff_candidates:
        ff_candidates = list(out_dir.glob("u_rc1*.nii"))
    if not ff_candidates:
        raise RuntimeError(f"Flow field no encontrado en {out_dir}")
    ff_path = ff_candidates[0]

    # Buscar mapa GM
    gm_candidates = list(out_dir.glob(f"mw*{stem[3:]}*.nii"))
    if not gm_candidates:
        gm_candidates = list(out_dir.glob("mw*.nii"))
    if not gm_candidates:
        raise RuntimeError(
            f"Mapa GM no encontrado en {out_dir} después de DARTEL existing.\n"
            f"Archivos disponibles: {[f.name for f in out_dir.iterdir()]}"
        )

    return gm_candidates[0], ff_path