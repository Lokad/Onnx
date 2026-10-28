"""Build staging and owned-process operations shared by routine and strict benchmarks."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import campaign_evidence as evidence

ROOT = Path(__file__).resolve().parent.parent
RUNNER = "Lokad.Onnx.Campaign.dll"

def save_json(path, value):
    # All output paths are newly owned files. Never overwrite prior evidence.
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

def read_build(prepared, role):
    path = prepared / role / "build.json"
    build = json.loads(path.read_text(encoding="utf-8-sig"))
    core = (path.parent / build["core"]).resolve()
    evidence.require(core.is_file(), "missing core: " + str(core))
    evidence.require(evidence.digest(build["core_sha256"], "built core") == evidence.sha256(core), "built core changed: " + role)
    evidence.require(evidence.digest(build["source_archive_sha256"], "source archive") == evidence.sha256(path.parent / "source.zip"), "source archive changed: " + role)
    evidence.digest(build["source_sha"], "source_sha", 40)
    evidence.require(isinstance(build["sdk"], str) and build["sdk"], "build SDK missing")
    return dict(build, path=core)

def stage(prepared, output, kind, runner_name=RUNNER):
    baseline = read_build(prepared, "L0")
    candidate = read_build(prepared, "L1") if kind == "comparison" else baseline
    evidence.require(baseline["sdk"] == candidate["sdk"], "core build SDKs differ")
    evidence.require(baseline.get("target_framework", "net10.0") == candidate.get("target_framework", "net10.0"),
                     "core target frameworks differ")
    layouts = [build.get("importer_layout", "integrated") for build in (baseline, candidate)]
    evidence.require(all(layout in ("integrated", "separate") for layout in layouts), "unknown importer assembly layout")
    evidence.require(layouts[0] == layouts[1], "integrated and separate importer assemblies require different runners; choose compatible source refs")
    runner = prepared / "runner"
    evidence.require((runner / runner_name).is_file(), "common runner missing")
    evidence.require(not output.exists(), "output already exists; choose a fresh directory")
    evidence.require(not output.is_relative_to(prepared), "output must be outside the prepared tree")
    output.mkdir(parents=True)
    result = {}
    for role, build in (("L0", baseline), ("L1", candidate)):
        destination = output / "staged" / role
        shutil.copytree(runner, destination)
        shutil.copyfile(build["path"], destination / "Lokad.Onnx.dll")
        result[role] = dict(build, runner=destination / runner_name)
    return result

def windows_affinity():
    import ctypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    kernel.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t)]
    kernel.GetProcessAffinityMask.restype = ctypes.c_int
    kernel.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    kernel.SetProcessAffinityMask.restype = ctypes.c_int
    handle = kernel.GetCurrentProcess()
    current, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not kernel.GetProcessAffinityMask(handle, ctypes.byref(current), ctypes.byref(system)):
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel, handle, current.value

def spawn_child(command, cpu, **kwargs):
    # Pin before the CLR starts: runtime workers must inherit the same CPU too.
    if sys.platform.startswith("linux"):
        return subprocess.Popen(["taskset", "--cpu-list", str(cpu), *command], **kwargs)
    evidence.require(sys.platform == "win32", "launch-time affinity requires Windows or Linux")
    import ctypes
    kernel, handle, original = windows_affinity()
    requested = 1 << cpu
    evidence.require(bool(original & requested), "requested CPU is outside the supervisor's allowed affinity")
    if not kernel.SetProcessAffinityMask(handle, requested):
        raise ctypes.WinError(ctypes.get_last_error())
    child = None
    try:
        # Windows children inherit the parent's affinity. Only this supervisor
        # is briefly narrowed, and its original mask is restored immediately.
        kwargs.setdefault("creationflags", subprocess.CREATE_NO_WINDOW)
        child = subprocess.Popen(command, **kwargs)
        return child
    finally:
        if not kernel.SetProcessAffinityMask(handle, original):
            failure = ctypes.WinError(ctypes.get_last_error())
            if child is not None and child.poll() is None:
                child.kill()
                child.wait()
            raise failure

def wait_child(child, label, timeout=3600):
    started = time.monotonic()
    try:
        while True:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError(label + " exceeded its process time limit")
            try:
                return child.wait(timeout=min(30, remaining))
            except subprocess.TimeoutExpired:
                print(label + " still running", flush=True)
    except BaseException:
        # This Popen object belongs only to the child we launched. Never search
        # for or terminate other agents' processes by name/PID enumeration.
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
