FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ---------------------------
# Core dependencies for Depth Anything V3
# ---------------------------
RUN pip install --no-cache-dir \
    "numpy<2.0" \
    scipy pillow \
    flask gunicorn \
    opencv-python-headless \
    huggingface_hub \
    "transformers>=4.30.0,<4.40.0" \
    timm \
    requests \
    tqdm \
    einops \
    omegaconf \
    moviepy

# ---------------------------
# Hugging Face Token Build Arg
# ---------------------------
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# ---------------------------
# Download Depth Anything V3 checkpoints
# ---------------------------
RUN python - <<EOF
import os
from huggingface_hub import hf_hub_download, snapshot_download

token = os.environ.get('HF_TOKEN', '')
os.makedirs('/opt/depth/checkpoints', exist_ok=True)

try:
    # Try downloading Depth Anything V2 (V3 may have different repo)
    snapshot_download(
        repo_id='depth-anything/Depth-Anything-V2-Large',
        local_dir='/opt/depth/checkpoints/depth-anything-v2',
        token=token if token else None
    )
    print("✓ Depth Anything checkpoint downloaded")
except Exception as e:
    print(f"Note: Will download on first use - {e}")
EOF

# ---------------------------
# Depth Anything V3 source code
# ---------------------------
COPY libs/Depth-Anything-3 /opt/depth/src
RUN pip install -e /opt/depth/src --no-deps || true

# ---------------------------
# Copy server file
# ---------------------------
COPY servers/depth_anything/server.py /app/server.py

# ---------------------------
# Python path
# ---------------------------
ENV PYTHONPATH="/opt/depth/src:${PYTHONPATH}"

EXPOSE 8002

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8002/health'); exit(0 if r.ok else 1)"

CMD ["python", "server.py"]