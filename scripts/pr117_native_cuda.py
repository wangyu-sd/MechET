"""Bind the host CUDA driver for an isolated PR117 vLLM allocation."""

from pathlib import Path
import re
import subprocess


CEPH = Path("/aaa/fionafyang/buddy1")
DRIVER_LIBRARIES = {
    "libcuda.so.1": "libcuda.so.",
    "libnvidia-ptxjitcompiler.so.1": "libnvidia-ptxjitcompiler.so.",
    "libnvidia-nvvm.so.4": "libnvidia-nvvm.so.",
}


def match_host_libraries(driver, catalog):
    """Resolve SONAMEs to libraries from the kernel driver's exact release."""
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", driver):
        raise ValueError("Invalid NVIDIA driver version from the scheduler-visible device")
    matches = {}
    for entry in catalog.splitlines():
        if " => " not in entry:
            continue
        description, path = entry.strip().split(" => ", 1)
        soname = description.split()[0]
        if soname not in DRIVER_LIBRARIES or "x86-64" not in description:
            continue
        resolved = Path(path).resolve()
        if resolved.name == DRIVER_LIBRARIES[soname] + driver and resolved.is_file():
            matches.setdefault(soname, resolved)
    if "libcuda.so.1" not in matches:
        raise ValueError(f"No x86-64 kernel-matching libcuda for NVIDIA driver {driver}")
    return matches


def bind_native_driver(runtime, output, gpu_uuid):
    output = output.resolve()
    if not output.is_relative_to(CEPH):
        raise ValueError("PR117 driver binding must reside on shared storage")
    driver = subprocess.check_output(["nvidia-smi", "-i", gpu_uuid, "--query-gpu=driver_version",
                                      "--format=csv,noheader"], text=True).strip()
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){1,2}", driver) or int(driver.split(".", 1)[0]) < 580:
        raise ValueError("PR117 native CUDA 12.9 mode requires an NVIDIA R580 or newer host driver")
    libraries = match_host_libraries(driver, subprocess.check_output(["ldconfig", "-p"], text=True))
    binding = output / "native-driver"
    binding.mkdir(exist_ok=False)
    for soname, path in libraries.items():
        (binding / soname).symlink_to(path)
    config = dict(runtime, cuda_compat=str(binding))
    receipt = dict(gpu_uuid=gpu_uuid, kernel_driver=driver, cuda_runtime="12.9",
                   libraries={name: str(path) for name, path in libraries.items()},
                   library_search_prefix=str(binding), mode="host_native_cuda")
    return config, receipt
