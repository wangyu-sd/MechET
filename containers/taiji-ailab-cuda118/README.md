# Taiji AILab CUDA 11.8 compatibility image

This image is the compatibility bridge for the ordinary
`AILab_AI4S_kemo_QY` A100 pool.

The pool currently exposes NVIDIA driver `450.80.02`.  The existing
`mirrors.tencent.com/whaleywang/metabo:taiji6` image has the required Ubuntu
22.04 / glibc 2.35 userspace, but declares `NVIDIA_REQUIRE_CUDA=cuda>=12.2`.
The NVIDIA OCI hook therefore rejects the container before it starts.  The
AILab-provided Ubuntu 18.04 image starts, but its glibc 2.27 cannot load the
project's `pyarrow` wheel, which requires glibc 2.28 or newer.

The compatibility image inherits `taiji6` and changes only the NVIDIA
admission requirement to CUDA 11.8.  MechET continues to use the frozen
Ceph-hosted `torch 2.6.0+cu118` Python environment.  A runtime preflight fails
closed unless glibc, CUDA, PyTorch, PyArrow, scikit-learn, Transformers, PEFT,
RDKit, and a CUDA matrix multiplication all work.

## Build and push

Run the build script from anywhere inside the checkout:

```bash
bash scripts/build_taiji_ailab_cuda118_image.sh
```

The default immutable tag is:

```text
mirrors.tencent.com/whaleywang/metabo:taiji6-cuda118-compat-v1
```

The script refuses a destination outside `mirrors.tencent.com`, uses
`linux/amd64`, sends only this small container directory as the build context,
streams build output to the terminal, and pushes directly with BuildKit.
Registry login must already be configured on the build host.

After pushing, run the one-card compatibility probe before changing an 8-card
experiment.  The probe must show:

- Ubuntu 22.04 and glibc 2.35 (or newer);
- A100 driver 450.80.02 visible through `nvidia-smi`;
- `torch.version.cuda == 11.8`;
- successful imports of PyArrow, scikit-learn, Transformers, PEFT, and RDKit;
- a successful CUDA matrix multiplication and Qwen3-8B load.

Do not skip the probe: overriding `NVIDIA_REQUIRE_CUDA` is safe only because
the actual experiment environment is pinned to CUDA 11.8.
