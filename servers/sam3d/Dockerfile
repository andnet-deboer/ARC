FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV OPEN3D_CPU_RENDERING=true

WORKDIR /app

# --------------------------------------------------
# Python dependencies - NO open3d for now
# --------------------------------------------------
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    scipy \
    pillow \
    trimesh \
    flask gunicorn \
    einops tqdm \
    opencv-python-headless \
    omegaconf hydra-core iopath \
    huggingface_hub \
    "transformers>=4.30.0,<4.40.0" \
    timm \
    requests \
    utils3d \
    seaborn \
    gradio \
    loguru

# --------------------------------------------------
# Try installing open3d - if it fails at runtime,
# we'll create a stub
# --------------------------------------------------
RUN pip install --no-cache-dir "open3d==0.18.0" || echo "Open3D install completed with warnings"

# --------------------------------------------------
# Hugging Face Token Build Arg
# --------------------------------------------------
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# --------------------------------------------------
# Download SAM-3D checkpoints
# --------------------------------------------------
RUN python - <<'EOF'
import os
from huggingface_hub import snapshot_download

token = os.environ.get('HF_TOKEN', '')
snapshot_download(
    repo_id='facebook/sam-3d-objects',
    local_dir='/opt/sam3d/checkpoints/hf-download',
    repo_type='model',
    token=token if token else None
)
EOF

RUN mv /opt/sam3d/checkpoints/hf-download/checkpoints /opt/sam3d/checkpoints/hf && \
    rm -rf /opt/sam3d/checkpoints/hf-download

# --------------------------------------------------
# MoGe (utils3d dependency)
# --------------------------------------------------
RUN python - <<'EOF'
import urllib.request, zipfile, shutil, os, glob

urllib.request.urlretrieve(
    "https://github.com/microsoft/MoGe/archive/refs/heads/main.zip",
    "/tmp/moge.zip"
)

with zipfile.ZipFile("/tmp/moge.zip") as z:
    z.extractall("/opt")

shutil.move("/opt/MoGe-main", "/opt/MoGe")
os.remove("/tmp/moge.zip")

for io_file in glob.glob("/opt/MoGe/**/io.py", recursive=True):
    os.rename(io_file, io_file.replace("/io.py", "/io_moge.py"))

for py_file in glob.glob("/opt/MoGe/**/*.py", recursive=True):
    try:
        with open(py_file, "r") as f:
            content = f.read()
        content = content.replace("from moge.utils.io import", "from moge.utils.io_moge import")
        content = content.replace("from .io import", "from .io_moge import")
        content = content.replace("import moge.utils.io", "import moge.utils.io_moge as io")
        with open(py_file, "w") as f:
            f.write(content)
    except:
        pass
EOF

RUN pip install -e /opt/MoGe --no-deps

# --------------------------------------------------
# SAM-3D-Objects source
# --------------------------------------------------
COPY libs/sam-3d-objects /opt/sam3d/src
RUN pip install -e /opt/sam3d/src --no-deps

# --------------------------------------------------
# Server
# --------------------------------------------------
COPY servers/sam3d/server.py /app/server.py

ENV PYTHONPATH="/opt/MoGe:/opt/MoGe/moge:/opt/MoGe/moge/utils:/opt/sam3d/src:/opt/sam3d/src/notebook:/opt/sam3d/src/sam3d_objects:${PYTHONPATH}"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/health'); exit(0 if r.ok else 1)"

CMD ["python", "server.py"]