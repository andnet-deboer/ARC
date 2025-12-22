# PyTorch image with Python + CUDA pre-installed
FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-devel

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# ============================================================
# Pin numpy FIRST
# ============================================================
RUN pip install --no-cache-dir "numpy==1.26.2"

# ============================================================
# PyTorch Geometric for torch 2.1.1
# ============================================================
RUN pip install --no-cache-dir \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

RUN pip install --no-cache-dir "torch-geometric==2.4.0"

# ============================================================
# Scientific stack
# ============================================================
RUN pip install --no-cache-dir \
    "scipy==1.11.4" \
    "scikit-learn==1.3.2" \
    "opencv-python-headless==4.9.0.80" \
    "Pillow==10.1.0" \
    "open3d==0.18.0" \
    "trimesh==4.0.4"

# ============================================================
# cgn-pytorch dependencies
# ============================================================
RUN pip install --no-cache-dir \
    "pyrender>=0.1.45,<0.2.0" \
    "meshcat==0.3.2" \
    "pyzmq" \
    "typeguard" \
    "importlib_resources" \
    tqdm \
    flask \
    gunicorn \
    requests

# ============================================================
# Install cgn-pytorch
# ============================================================
RUN pip install --no-cache-dir --no-deps cgn-pytorch==0.4.3

# ============================================================
# Force numpy version 
# ============================================================
RUN pip install --no-cache-dir "numpy==1.26.2"

# ============================================================
# Copy server file
# ============================================================
COPY servers/contact_graspnet/server.py /app/server.py

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; r = requests.get('http://localhost:8003/health'); exit(0 if r.ok else 1)"

CMD ["python", "server.py"]