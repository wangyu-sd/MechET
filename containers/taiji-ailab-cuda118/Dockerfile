# syntax=docker/dockerfile:1.7

# taiji6 already provides the Ubuntu 22.04 / glibc 2.35 userspace used by the
# successful MechET jobs.  Its inherited NVIDIA_REQUIRE_CUDA value requires
# CUDA 12.2, however, so the NVIDIA OCI hook rejects AILab hosts with driver
# 450.80 before the container can start.  MechET actually runs the Ceph-hosted
# torch 2.6.0+cu118 environment, for which the CUDA 11.x compatibility floor is
# sufficient on those hosts.
ARG BASE_IMAGE=mirrors.tencent.com/whaleywang/metabo:taiji6
FROM ${BASE_IMAGE}

USER root

# This is an OCI admission contract, not a replacement of CUDA libraries.
# Runtime validation below requires the actual Python environment to report
# torch.version.cuda == 11.8 before an experiment may start.
ENV NVIDIA_REQUIRE_CUDA="cuda>=11.8" \
    NVIDIA_DRIVER_CAPABILITIES="compute,utility" \
    MECHET_IMAGE_CONTRACT="ailab-a100-driver450-cu118-glibc235-v1" \
    PYTHONUNBUFFERED="1"

COPY image_preflight.sh /usr/local/bin/mechet-ailab-image-preflight

RUN chmod 0755 /usr/local/bin/mechet-ailab-image-preflight \
    && /usr/local/bin/mechet-ailab-image-preflight --build

WORKDIR /workspace
CMD ["/bin/bash"]
