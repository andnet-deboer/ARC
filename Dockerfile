# ---------------------------
# Base image
# ---------------------------
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# ---------------------------
# Python dependencies
# ---------------------------
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    scipy pillow \
    trimesh open3d \
    flask gunicorn \
    einops tqdm \
    opencv-python-headless \
    omegaconf hydra-core iopath \
    huggingface_hub \
    transformers \
    requests

# ---------------------------
# Hugging Face Token Build Arg
# ---------------------------
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# ---------------------------
# Hugging Face download SAM-3D checkpoints
# ---------------------------
RUN python - <<EOF
import os
from huggingface_hub import snapshot_download

os.environ['HF_TOKEN'] = '$HF_TOKEN'

snapshot_download(
    repo_id='facebook/sam-3d-objects',
    local_dir='/opt/sam3d/checkpoints/hf-download',
    repo_type='model'
)
EOF

RUN mv /opt/sam3d/checkpoints/hf-download/checkpoints /opt/sam3d/checkpoints/hf && \
    rm -rf /opt/sam3d/checkpoints/hf-download

# ---------------------------
# SAM-3D-Objects source code
# ---------------------------
RUN python - <<'EOF'
import urllib.request, zipfile, shutil, os

# Download SAM3D repo
urllib.request.urlretrieve(
    "https://github.com/facebookresearch/sam-3d-objects/archive/refs/heads/main.zip",
    "/tmp/sam3d.zip"
)
with zipfile.ZipFile("/tmp/sam3d.zip") as z:
    z.extractall("/opt")
shutil.move("/opt/sam-3d-objects-main", "/opt/sam3d")
os.remove("/tmp/sam3d.zip")

# Install SAM3D in editable mode
os.system("pip install -e /opt/sam3d --no-deps")
EOF

# ---------------------------
# SAM3 (2D) source code
# ---------------------------
RUN python - <<'EOF'
import urllib.request, zipfile, shutil, os

# Download SAM3 repo
urllib.request.urlretrieve(
    "https://github.com/facebookresearch/sam3/archive/refs/heads/main.zip",
    "/tmp/sam3_2d.zip"
)
with zipfile.ZipFile("/tmp/sam3_2d.zip") as z:
    z.extractall("/opt")
shutil.move("/opt/sam3-main", "/opt/sam3")
os.remove("/tmp/sam3_2d.zip")

# Install SAM3 in editable mode
os.system("pip install -e /opt/sam3 --no-deps")
EOF

# ---------------------------
# Depth Anything 3 source code (ByteDance official repo)
# ---------------------------
RUN python - <<'EOF'
import urllib.request, zipfile, shutil, os, glob

# Download Depth Anything 3 repo (using ByteDance repo which is the official one)
urllib.request.urlretrieve(
    "https://github.com/ByteDance-Seed/depth-anything-3/archive/refs/heads/main.zip",
    "/tmp/depth_anything_v3.zip"
)
with zipfile.ZipFile("/tmp/depth_anything_v3.zip") as z:
    z.extractall("/tmp")

# Find the extracted directory (it might have a different name)
extracted_dirs = glob.glob("/tmp/depth-anything-3*")
if extracted_dirs:
    extracted_dir = extracted_dirs[0]
    shutil.move(extracted_dir, "/opt/depth_anything_v3")
    print(f"Moved {extracted_dir} to /opt/depth_anything_v3")

os.remove("/tmp/depth_anything_v3.zip")

# Install Depth Anything V3 in editable mode
os.system("pip install -e /opt/depth_anything_v3 --no-deps")
EOF

# ---------------------------
# MoGe (provides utils3d)
# ---------------------------
RUN python - <<'EOF'
import urllib.request, zipfile, shutil, os

# Download MoGe repo
urllib.request.urlretrieve(
    "https://github.com/microsoft/MoGe/archive/refs/heads/main.zip",
    "/tmp/moge.zip"
)
with zipfile.ZipFile("/tmp/moge.zip") as z:
    z.extractall("/opt")
shutil.move("/opt/MoGe-main", "/opt/MoGe")
os.remove("/tmp/moge.zip")

# Install MoGe in editable mode
os.system("pip install -e /opt/MoGe --no-deps")
EOF

# ---------------------------
# Server
# ---------------------------
WORKDIR /app
COPY sam3d_server.py /app/
COPY sam3_server.py /app/
COPY depth_anything_v3_server.py /app/

# ---------------------------
# Python path
# ---------------------------
ENV PYTHONPATH="/opt/sam3d:/opt/sam3d/sam-3d-objects-main:/opt/sam3d/sam-3d-objects-main/notebook:/opt/sam3d/sam-3d-objects-main/sam3d_objects:/opt/sam3:/opt/sam3/sam3:/opt/depth_anything_v3:/opt/depth_anything_v3/depth_anything_v3:/opt/MoGe/moge:/opt/MoGe/moge/utils:${PYTHONPATH}"

# ---------------------------
# Expose ports
# ---------------------------
EXPOSE 8000 8001 8002

# ---------------------------
# Start all 3 servers
# ---------------------------
CMD ["bash", "-c", "python sam3d_server.py & python sam3_server.py & python depth_anything_v3_server.py & wait"]
