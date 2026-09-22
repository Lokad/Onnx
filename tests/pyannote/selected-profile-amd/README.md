# Refresh selected Pyannote attribution on AMD

Use the exact selected Core `1279b4b6` / Data `4e602d9f`. The preceding Parakeet
arithmetic candidate failed its fixed speed gates and is not used here. The
native layout diagnostic is closed before this fresh managed capture starts.

Reuse the independently qualified Linux consumer. Change only its two expected
product hash literals. Compiled inspection requires 160 existing methods,
159 unchanged, one changed main method, no added/removed methods and unchanged
public declarations. All request, thread, ownership and numerical checks stay
the same. Product binaries themselves are copied exactly, without rebuilding.

The previous closed AMD protocol supplies one control and two sampled processes.
Each runs four original fixtures with one warmup and three measured passes:
48 public requests. Compare every output to the currently selected AMD result.
Preserve the post-warmup barrier, same-thread markers, all event reconciliation
and coverage limits, and both Speedscope and Chromium exports. Sampled thread
weights, wall latency and process CPU remain separate. No speed selection or
new Microsoft ORT ratio is inferred from profiling.

Run from repository root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/selected-profile-amd/prepare.py
    tests/pyannote/selected-profile-amd/transport.py stage
    tests/pyannote/selected-profile-amd/transport.py launch
    tests/pyannote/selected-profile-amd/transport.py observe
    tests/pyannote/selected-profile-amd/transport.py collect
    tests/pyannote/selected-profile-amd/export.py
    tests/pyannote/selected-profile-amd/audit.py

All owners must be actually terminal before collection. Never relaunch to poll
or after an SSH timeout. Refuse existing destinations and preserve failures.
Local output is `artifacts/pyannote-selected-profile-amd-20260922`; remote output
is `/dev/shm/lokad-pyannote-selected-profile-20260922`.

Keep CPU2 for target and CPU0 for collector/monitor, before runtime startup.
Preflight requires 10 GiB available and 3 GiB tmpfs; each pair retains 900-second,
8 GiB RSS, 1 GiB available/tmpfs/output and 2 GiB experiment bounds. The release
barrier is 180 seconds. Exact PID/birth ownership remains mandatory; only the
consumer's wall-clock metadata permits the original 1.1-second Linux rounding.
Local builds/exports retain the original 8 GiB preflight/RSS and resource checks.
