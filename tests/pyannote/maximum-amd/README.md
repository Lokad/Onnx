# Maximum-duration diarization on AMD

This lane qualifies the public `Community1Diarizer.Diarize` API on the existing
600-second repeated-dialogue input. It links the already qualified product
`8732831` DLLs without rebuilding or modifying the product. The earlier Windows
maximum request used `21f3e74`; the diarization and embedding Data source is
unchanged between these builds.

The input is twenty exact copies of the pinned thirty-second dialogue,
reconstructed in memory to avoid duplicating a 38.4 MB PCM file on the VM.
Preparation verifies the original long receipt, all 4,146 reproduced native
arrays, the PCM recipe and model/product hashes. The runner is a checked
derivation of the original maximum/recovery probe, with an explicit scope
label and environment reporting. A new small `Evidence.cs` binds the exact
short manifest, PCM, models and core/Data DLLs before inference.

The acceptance is a complete 600-second, 591-window call; native speaker
timelines and centroid agreement; two refusals; empty input; short native-bound
recovery; and exact retained input/output ownership. Native timestamp boundaries
retain `1e-12` tolerance and scaled centroid error retains `1e-4`. The stable
native tie policy and its 98 differences from original unstable exclusive
selection remain explicit. Repeated speech is a resource test, not independent
human-labeled accuracy. No full managed long intermediate trace is captured.

Use the existing Python environment under
`artifacts/asr-labeled-20260919/venv/Scripts/python.exe` from the repository root.
Every writer refuses an existing destination:

    python -B tests/pyannote/maximum-amd/prepare.py --artifact <new-directory>
    python -B tests/pyannote/maximum-amd/vm.py deploy --artifact <directory>
    python -B tests/pyannote/maximum-amd/vm.py poll --artifact <directory>
    python -B tests/pyannote/maximum-amd/vm.py collect --artifact <directory>
    python -B <directory>/payload/audit.py --artifact <directory> --output <directory>/audit.json

The fixed recorded artifact is `artifacts/pyannote-maximum-amd-20260919`.
Its preparation/build has completed; do not repeat it into that directory.
Deployment requires the preceding Whisper AMD job to be terminal and collected.
All four models already exist under earlier artifact directories on the VM;
matching binaries and the thirty-second PCM are reused by verified hard links.
The transferred payload is digest-verified before extraction and model loading.

The worker inherits CPU 2 before startup; supervision uses CPU 0 and enforces
eight GiB process-group RSS, 1,800 seconds and 256 MiB available memory. No GC or
experimental runtime switches are forced. Each sample records actual process
births, affinity, CPU consumption, RSS and system available memory. Collection
requires every observed identity to be absent.

After the independent audit passes, run `close_report.py close --artifact
<directory>` once, followed by `close_report.py report --artifact <directory>`
once. Those commands retain all identities and create the report and observations.
Malformed-evidence tests use the previously closed Windows/native fixtures:

    python -B -m unittest discover -s tests/pyannote/maximum-amd -p test_audit.py -v

The API stopwatch includes features, inference, clustering and owned output
construction. Native reference generation includes validation and array export
and supplies no comparable ORT inference timing. Dedicated matched application
measurements are in [BENCHMARK.md](../../../BENCHMARK.md#audio-matched-microsoft-onnx-runtime-baselines).
