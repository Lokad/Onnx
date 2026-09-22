# M23 complete numerical qualification on AMD

Bind the four already qualified M21 consumers to the isolated M23 Core. Only
their embedded Core hash changes. Every compiled consumer method is compared;
exactly the Main hash literal may differ, with all other instructions and the
public surface preserved. Raw/wide/span each contain 145 methods; layers 134.

Run `run.py prepare`, `stage`, `launch`, `observe`, `collect`, then `audit.py`
with Python 3.13 `-X utf8 -B` from the repository root. Destinations are
`artifacts/pyannote-kernel-loop-numerics-amd-20260922` and
`/dev/shm/lokad-pyannote-kernel-loop-numerics-20260922`. Never overwrite a run.

Twelve restore/build/inventory jobs precede eight numerical workers. Each of
raw, wide, span and layers runs with `DOTNET_EnableAVX512=0`, then ordinary
AVX512. Each raw family keeps 2,648 cases, 20 supplemental and 10 rejections,
2,668 control requests and 5,336 prepared requests. Captured layers keep all
108 graphs, 119,823,360 values and original native bounds. Read-only operands,
held outputs, sentinels, scratch accounting and fallback checks remain.

Worker CPU2, monitor CPU0; 12 GiB available/3 GiB tmpfs before each job,
8 GiB owned RSS, 900 seconds per job, 1 GiB minimum available/tmpfs and output
bound, 2 GiB total artifacts. All inputs, resources and process birth identities
are retained. No performance worker runs here. Successful closure permits
generated-code inspection, followed by the prospective fixed component screen.
