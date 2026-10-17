# Gaudi2 single-card vLLM service image (docs/DOCKER.md).
#
#   docker build -f docker/Dockerfile -t gaudi2-single-card-vllm:latest .
#
# Base: Intel's PyTorch image for Gaudi SDK 1.24.1 (torch 2.11.0a0, habana-torch-plugin 1.24.1.482), the exact
# stack this project was built and measured on. The host needs the matching Gaudi driver (1.24.1); nothing else
# from the host is used except the device nodes, the model directories and a data directory for caches and logs.
ARG BASE_IMAGE=vault.habana.ai/gaudi-docker/1.24.1/ubuntu24.04/habanalabs/pytorch-installer-2.11.0:latest
FROM ${BASE_IMAGE}

ARG PIP_INDEX_URL=https://pypi.org/simple
# optional Ubuntu archive mirror, e.g. https://mirrors.aliyun.com/ubuntu (empty = the image's default)
ARG APT_MIRROR=
# the TPC compiler (tpc-clang) for the custom kernels; the measured build used this version
ARG HABANALABS_TOOLS_VERSION=1.24.2-481
ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONNOUSERSITE=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# build tools for the native kernels (tpc-clang from Intel's Gaudi apt repository), tcmalloc (LD_PRELOAD in serve.sh)
RUN if [ -n "$APT_MIRROR" ]; then sed -i "s#http://archive.ubuntu.com/ubuntu/\?#$APT_MIRROR#; s#http://security.ubuntu.com/ubuntu/\?#$APT_MIRROR#" /etc/apt/sources.list.d/ubuntu.sources; fi \
 && apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl gnupg \
 && curl -fsSL https://vault.habana.ai/artifactory/api/gpg/key/public | gpg --dearmor -o /usr/share/keyrings/habana.gpg \
 && echo "deb [signed-by=/usr/share/keyrings/habana.gpg] https://vault.habana.ai/artifactory/debian noble main" \
      > /etc/apt/sources.list.d/habanalabs.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake libgoogle-perftools4t64 \
      habanalabs-tools=${HABANALABS_TOOLS_VERSION} \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/gaudi2-single-card-vllm
COPY . .

# Python packages into the image's python3.12 (editable: the sources in this directory are what runs), then the
# TPC kernels, PyTorch ops and the kernel-library front-end. Build directories are dropped, only the libraries stay.
RUN rm -f .python && PYTHON=python3.12 scripts/install.sh \
 && MAX_JOBS=${MAX_JOBS} scripts/build_native.sh \
 && find build/native -mindepth 1 -maxdepth 1 -type d -exec rm -rf {} + \
 && find . -name __pycache__ -prune -exec rm -rf {} + \
 && cd / && VLLM_PLUGINS=none TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 \
      PYTHONPATH=/opt/gaudi2-single-card-vllm/vllm-gaudi python3.12 -c \
      "import vllm.entrypoints.openai.api_server, vllm.entrypoints.cli.main, vllm_gaudi, flashinfer_gaudi, dflash2_request_guard; print('imports ok')"

ENV PIP_INDEX_URL=
EXPOSE 8000
ENTRYPOINT ["/opt/gaudi2-single-card-vllm/docker/entrypoint.sh"]
