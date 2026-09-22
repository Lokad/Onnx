# Confirm Pyannote's native convolution layout on AMD

This diagnostic follows the closed Parakeet comparison. It refuses incomplete
or live preceding owners. It constructs both optimized models in one process,
then profiles six requests per model in a separate process: three retained
graph inputs, twice each. No whole-application speed claim is made.

Keep ORT 1.29.0, CPU2 before imports, one intra/inter-op thread, sequential
execution, full optimization and no spinning. Pin the exact native packages,
interpreter, model bytes, six inputs and six original native reference arrays.
Require the unchanged `1e-4` scaled bound, finite arrays, exact repeats, unchanged
inputs and held outputs. Resolve every executed profile node to its optimized
graph, including nested branches. Every top-level node must execute six times.
Serialize all nodes, domains, attributes, weight shapes and layout conversions.
Parse collected graphs on the workstation with its separately pinned ONNX/NumPy
packages. The VM needs only its existing ORT and NumPy; no parser is installed.

From repository root, use `C:/Python313/python.exe -X utf8 -B`:

    -m unittest discover -s tests/pyannote/native-layout-amd -p test_*.py -v
    tests/pyannote/native-layout-amd/run.py prepare
    tests/pyannote/native-layout-amd/run.py stage
    tests/pyannote/native-layout-amd/run.py launch
    tests/pyannote/native-layout-amd/run.py observe
    tests/pyannote/native-layout-amd/run.py collect
    tests/pyannote/native-layout-amd/audit.py

Preparation and launch refuse existing output. Observe the existing PID/birth
after a timeout; never restart it to poll. Collect only after actual termination.
All tools and inputs freeze before preparation. Preserve any failure and use an
explicit successor for corrections. The independent audit rechecks all arrays,
graph/profile coverage, resource samples and collected hashes.

Use 12 GiB available and 3 GiB tmpfs preflight, 8 GiB owned RSS, 1 GiB available
and tmpfs floors, 1 GiB per-worker output, 2 GiB artifacts and 900-second workers.
Monitor CPU0 and retain every preflight observation. Only owned descendants may
be stopped on a failed bound. No product mutation, download, push or publication.

Local output: `artifacts/pyannote-native-layout-amd-20260922`.
Remote output: `/dev/shm/lokad-pyannote-native-layout-20260922`.
