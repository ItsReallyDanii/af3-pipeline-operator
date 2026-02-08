# Base: NVIDIA PyTorch (CUDA-enabled)
FROM nvcr.io/nvidia/pytorch:24.01-py3 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --- STAGE 1: System Deps ---
RUN apt-get update && apt-get install -y \
    git \
    wget \
    kalign \
    aria2 \
    && rm -rf /var/lib/apt/lists/*

# --- STAGE 2: OpenFold-3 (Pinned example SHA) ---
WORKDIR /opt/openfold3
RUN git clone https://github.com/aqlaboratory/openfold-3.git . \
    && git checkout aeac5fd

# Try extras first, fallback to base install
RUN pip install --no-cache-dir .[cuequivariance] || pip install --no-cache-dir .

# --- STAGE 3: AQAffinity (Pinned source channel) ---
RUN pip install --no-cache-dir \
    "git+https://huggingface.co/SandboxAQ/aqaffinity@main"

# --- STAGE 4: Pipeline Runtime ---
WORKDIR /app/pipeline
COPY requirements.txt /app/pipeline/requirements.txt
RUN pip install --no-cache-dir -r /app/pipeline/requirements.txt

ENTRYPOINT ["python", "scripts/run_pipeline.py"]
