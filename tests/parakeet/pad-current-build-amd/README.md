# Existing padding dispatcher on the qualified current product

One candidate, selected by the completed ordinary-runtime warmup diagnostic
4402da8e. Apply the retained M47 helper and six independent tests byte-for-byte to
the qualified root source (435 original inputs, 437 candidate inputs). Root
product files stay unchanged until complete qualification. No new compiler flag,
kernel variant, model copy or performance measurement belongs to this build.

Reuse the earlier eight-job build supervisor, flags-aware inspector and six-test
census. SDK 10.0.204; CLI restore/build, backend restore/build, Core/Data inventory,
then all six public Pad tests normally and with AVX512 disabled. The current
selected pair is Core f3992f40 / Data a8e0b583. Require all 3,281 existing Core
methods and 697 Data methods to remain exact except the four resolved public Pad
call targets. Original PadCore body/flags, public surface and generated names
stay exact. The one private helper's complete normalized compiled body must also
equal the previously reviewed M47 helper, not just have its name.

The independent tests cover all four types, exceptional float bit patterns,
ranks 1–5, empty dimensions, asymmetric padding, reversed/sliced/broadcast input,
cropping, reflection and ownership. Passing this focused build does not admit
the candidate. Fixed-warmup repeated-process screening and full Parakeet/shared-
model qualification remain required.

Keep the existing build bounds: 10 GiB available RAM and 3 GiB free tmpfs before
each job; 8 GiB owned RSS; 1 GiB remaining RAM/tmpfs; 1 GiB output/job; 2 GiB
total; 900 seconds/job and four hours overall. CPU 2 builds/tests, CPU 0 monitors.
Use the existing offline feed; no network restore or Windows .NET build.

Run Python with `C:/Python313/python.exe -X utf8 -B`. First execute
`tests/parakeet/pad-current-source/prepare.py` from the root. From this directory,
run `test_checks.py`, then `run.py` with `prepare`, `stage`, `launch`, `observe`,
`collect` as separate actions, then `audit.py`. Freeze tools and source at
preparation. Namespace: artifacts/parakeet-pad-current-build-amd-20260926 locally,
/dev/shm/lokad-parakeet-pad-current-build-20260926 on the authorized VM.
Do not repeat a completed job or alter a prepared gate after seeing results.
