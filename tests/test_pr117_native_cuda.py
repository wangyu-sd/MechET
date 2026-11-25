from pathlib import Path

import pytest

from scripts import pr117_native_cuda as native


def test_native_driver_uses_only_libraries_matching_the_kernel(tmp_path):
    driver = "580.105.08"
    old = tmp_path / "libcuda.so.575.57.08"
    direct = tmp_path / f"libcuda.so.{driver}"
    compiler = tmp_path / f"libnvidia-ptxjitcompiler.so.{driver}"
    for path in (old, direct, compiler):
        path.touch()
    catalog = f"""libcuda.so.1 (libc6,x86-64) => {old}
libcuda.so.1 (libc6,x86-64) => {direct}
libnvidia-ptxjitcompiler.so.1 (libc6,x86-64) => {compiler}"""
    assert native.match_host_libraries(driver, catalog) == {
        "libcuda.so.1": direct, "libnvidia-ptxjitcompiler.so.1": compiler,
    }


def test_missing_kernel_match_rejects_compat_library_without_staging_it(tmp_path):
    old = tmp_path / "libcuda.so.575.57.08"
    old.touch()
    with pytest.raises(ValueError, match="kernel-matching libcuda"):
        native.match_host_libraries("580.105.08", f"libcuda.so.1 (libc6,x86-64) => {old}")


def test_explicit_native_cuda_binding_preserves_runtime_fields(tmp_path, monkeypatch):
    driver = "580.105.08"
    library = tmp_path / f"libcuda.so.{driver}"
    library.touch()
    seen = []

    def query(command, *, text):
        seen.append(command)
        return f"{driver}\n" if command[0] == "nvidia-smi" else f"libcuda.so.1 (libc6,x86-64) => {library}\n"

    monkeypatch.setattr(native.subprocess, "check_output", query)
    monkeypatch.setattr(native, "CEPH", tmp_path)
    original = dict(cache_root=str(tmp_path / "old"), cuda_compat=str(tmp_path / "compat"), python="/some/python")
    replacement, receipt = native.bind_native_driver(original, tmp_path, "GPU-isolated")
    assert Path(replacement["cuda_compat"], "libcuda.so.1").resolve() == library
    assert original["cuda_compat"] != replacement["cuda_compat"]
    assert original["python"] == replacement["python"]
    assert receipt["kernel_driver"] == driver
    assert seen[0][1:3] == ["-i", "GPU-isolated"]
