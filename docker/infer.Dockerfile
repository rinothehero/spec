ARG BASE_IMAGE=nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
FROM ${BASE_IMAGE}

ARG VCS_REF=""
ARG IMAGE_NAME="xronos:latest"

LABEL org.opencontainers.image.title="Xronos speculative decoding inference"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.ref.name="${IMAGE_NAME}"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV XRONOS_GIT_COMMIT=${VCS_REF}
ENV XRONOS_IMAGE=${IMAGE_NAME}
ENV HF_HOME=/models/huggingface
ENV TRANSFORMERS_CACHE=/models/huggingface/transformers

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        git \
        python3 \
        python3-dev \
        python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/xronos

RUN mkdir -p /models/huggingface/transformers

COPY pyproject.toml README.md Makefile ./
COPY xronos ./xronos
COPY k8s ./k8s

RUN python3 -m pip install --upgrade pip setuptools wheel \
    && python3 -m pip install -e ".[infer]"

CMD ["python3", "-m", "xronos.infer.experiment_doctor", "--help"]
