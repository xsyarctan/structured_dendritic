FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3 \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/structured-dendrite

COPY pyproject.toml requirements.txt README.md train.py ./
COPY conf ./conf
COPY structured_dendrite ./structured_dendrite

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -e .

CMD ["python3", "train.py", "experiment=liq_ssm/listops_rs"]
