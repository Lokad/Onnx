# Bind the existing profiler to the current Pyannote product

Build only SampledAudio, with the two expected product hashes changed to the
qualified M22 Core/Data. All models, request checks, barriers, thread markers
and diagnostic logic stay intact. Reuse the prior Bridge binary to compare
all 160 consumer methods: 159 unchanged, only Main's two hash operands changed,
no added/removed methods and an equal public surface. Product binaries are
copied exactly and never rebuilt.

From repository root with Python 3.13 `-X utf8 -B`, run `test_checks.py`, then
`run.py prepare`, `stage`, `launch`, `observe`, `collect`, and `audit.py`.
Fresh destinations: `artifacts/pyannote-current-profile-build-amd-v2-20260922`
and `/dev/shm/lokad-pyannote-current-profile-build-v2-20260922`.

Four bounded AMD jobs identify SDK 10.0.204, restore, build and compare the
consumer. Use complete dependency sets for both assemblies. The existing
offline feed and SDK are pinned. Worker CPU2, monitor CPU0; 12 GiB available,
3 GiB tmpfs preflight, 8 GiB owned RSS, 900 seconds per job, 1 GiB minimum
available/tmpfs/output bound, 2 GiB artifacts. No capture or speed result is
produced here. Preserve every failed attempt and refuse existing destinations.

This successor preserves the first attempt, which stopped at SDK identity
because the isolated source lacked global.json and selected an installed preview.
It copies the repository global.json unchanged before any worker. No compilation
or inference occurred in the failed attempt. Expected SDK remains10.0.204.
