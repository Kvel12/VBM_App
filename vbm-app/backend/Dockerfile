# ─────────────────────────────────────────────────────────────────────────────
# VBM App — Backend Dockerfile
# Base: Ubuntu 22.04 LTS (compatible con MATLAB Runtime R2019b)
# Incluye: SPM12 Standalone r7771, MATLAB Runtime v97, ROBEX, Python 3.11
#
# Versión SPM12 r7771 fijada para compatibilidad bit-a-bit con los mapas de
# materia gris usados durante el entrenamiento de los modelos CNN/SVM.
#
# CPU por defecto. Si el host tiene GPU NVIDIA y nvidia-container-toolkit
# instalado, Docker detecta la GPU automáticamente vía torch.cuda.is_available()
# sin cambios en este Dockerfile.
# ─────────────────────────────────────────────────────────────────────────────
FROM ubuntu:22.04

# Evitar prompts interactivos de apt
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Bogota
ENV LD_LIBRARY_PATH=""

# ─── Dependencias del sistema ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Herramientas base
    wget curl unzip ca-certificates \
    # MATLAB Runtime R2019b requiere estas librerías
    libxt6 libxmu6 libgl1 libglu1-mesa \
    libxrender1 libxtst6 libxi6 \
    # Para NIfTI / nibabel
    zlib1g \
    # Python 3.11
    python3.11 python3.11-dev python3-pip \
    # ROBEX
    bc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Python por defecto → python3.11
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 \
 && ln -sf /usr/bin/python3.11 /usr/bin/python

# ─── MATLAB Runtime R2019b (v97) ──────────────────────────────────────────────
# Requerido por SPM12 Standalone r7771 (compilado con MATLAB R2019b).
# Tamaño: ~2.5 GB — se cachea en la capa de Docker.
ENV MCR_VERSION=R2019b
ENV MCR_DIR=/opt/mcr/v97

RUN mkdir -p /tmp/mcr_install && \
    wget -q --show-progress \
        "https://ssd.mathworks.com/supportfiles/downloads/R2019b/Release/9/deployment_files/installer/complete/glnxa64/MATLAB_Runtime_R2019b_Update_9_glnxa64.zip" \
        -O /tmp/mcr_install/mcr.zip && \
    unzip -q /tmp/mcr_install/mcr.zip -d /tmp/mcr_install && \
    /tmp/mcr_install/install -mode silent -agreeToLicense yes \
        -destinationFolder /opt/mcr && \
    rm -rf /tmp/mcr_install

# Variables de entorno requeridas por MCR
ENV LD_LIBRARY_PATH="${MCR_DIR}/runtime/glnxa64:${MCR_DIR}/bin/glnxa64:${MCR_DIR}/sys/os/glnxa64"
ENV XAPPLRESDIR="${MCR_DIR}/X11/app-defaults"

# ─── SPM12 Standalone r7771 ───────────────────────────────────────────────────
# Distribuido por UCL en: https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/dev/
# El zip se descarga manualmente y se incluye en backend/vendor/spm12_r7771.zip
# (también se puede regenerar con:
#  curl -L https://www.fil.ion.ucl.ac.uk/spm/download/restricted/utopia/dev/spm12_r7771_Linux_R2019b.zip
#       -o backend/vendor/spm12_r7771.zip)
#
# Estructura del zip:
#   spm12/
#     ├── spm12            (binario)
#     ├── run_spm12.sh     (launcher: ./run_spm12.sh <mcr_dir> <args>)
#     └── spm12.ctf
ENV SPM_STANDALONE_DIR=/opt/spm12_standalone

COPY vendor/spm12_r7771.zip /tmp/spm12_r7771.zip
RUN mkdir -p ${SPM_STANDALONE_DIR} && \
    unzip -q /tmp/spm12_r7771.zip -d /tmp/spm_extract && \
    mv /tmp/spm_extract/spm12/* ${SPM_STANDALONE_DIR}/ && \
    chmod +x ${SPM_STANDALONE_DIR}/run_spm12.sh && \
    chmod +x ${SPM_STANDALONE_DIR}/spm12 && \
    rm -rf /tmp/spm_extract /tmp/spm12_r7771.zip

# ─── ROBEX (skull stripping) ──────────────────────────────────────────────────
# NITRC sirve el binario detrás de una página de "agreement" (no se puede
# descargar con wget directo). Hay que aceptar el acuerdo en el navegador y
# guardar el tar.gz en backend/vendor/ROBEXv12.linux64.tar.gz
#
# 1. Ir a:  https://www.nitrc.org/projects/robex/
# 2. Pestaña "Files" → ROBEX v1.2 → Linux64 → "I agree"
# 3. Guardar como backend/vendor/ROBEXv12.linux64.tar.gz
#
# El tar extrae a /opt/ROBEX/ (top-level dentro del archivo).
ENV ROBEX_DIR=/opt/ROBEX

COPY vendor/ROBEXv12.linux64.tar.gz /tmp/robex.tar.gz
RUN tar -xzf /tmp/robex.tar.gz -C /opt && \
    chmod +x ${ROBEX_DIR}/runROBEX.sh && \
    chmod +x ${ROBEX_DIR}/ROBEX && \
    rm /tmp/robex.tar.gz

# ─── Python: instalar dependencias ───────────────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# PyTorch CPU base — si hay GPU disponible en runtime, torch.cuda.is_available()
# retorna True automáticamente sin reinstalar (el binario CPU soporta CUDA vía
# nvidia-container-toolkit montado en runtime).
# Para GPU dedicada: cambiar la línea de torch en requirements.txt a cu121.
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Código de la aplicación ──────────────────────────────────────────────────
COPY app/ ./app/

# Assets pesados: se montan como volúmenes en docker-compose.yml
# No se copian al contenedor para mantener la imagen portable
# - dartel_templates/  → /app/dartel_templates
# - models/            → /app/models

# ─── Directorio temporal de procesamiento ─────────────────────────────────────
RUN mkdir -p /app/tmp && chmod 777 /app/tmp

# ─── Entrypoint ───────────────────────────────────────────────────────────────
EXPOSE 8000

# --workers 1: SPM Standalone no es thread-safe entre workers
# --timeout-keep-alive 300: jobs pueden tardar varios minutos
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--timeout-keep-alive", "300"]
