# Actual Parakeet attention slice layouts

Confirm the exact tensors reaching `TensorSlice.Reshape` for all 24 attention
`Slice_1` outputs. The prior matched profile found 2.962s in complete Slice/Reshape
pairs versus ORT's 0.024s. This diagnostic records actual layouts before choosing
the guarded materialization implementation; it is not a performance candidate.

An isolated selected Core adds one observation call before the original Reshape
body, with unchanged method flags. A new internal probe records scalar metadata
only. Data stays byte-identical. The original complete application consumer gains
initialization, saving after its request clock, and an explicit diagnostic Core
identity check. It preserves original warmup/measurement passes and all public,
native, repeat, input and held-output checks: 80 requests on all twenty clips.
No tensors or execution contexts survive their original lifetimes.

Build and capture are separate bounded stages, with compiled scope review before
inference. Build limits are 2 GiB available/1 GiB tmpfs at start, 3 GiB owned RSS
and 180 seconds per command. Capture requires 11 GiB available/2 GiB tmpfs, caps
owned RSS at 12 GiB and duration at 900 seconds. Both retain 1 GiB memory/tmpfs,
cap stage output at 512 MiB, use CPU2 for work and CPU0 for monitoring, and record
all owner PID/birth identities. Reuse the previous bounded process supervisor
as a frozen `common.py` in the payload; its source hash is recorded.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root with `run.py
prepare`, `stage`, `launch build`, `observe build`, `collect build`, then
`review_build.py`; only an admitted review permits `launch capture`. Observe
until terminal, collect, then run `audit.py`. Every namespace is exclusive-create.
Never reopen a closed stage or repeat inference to fix an analyzer.

Local: `artifacts/parakeet-slice-layout-amd-20260924`.
VM: `/dev/shm/lokad-parakeet-slice-layout-20260924`.
The active ExecPlan is `.agent/m65-parakeet-ort-diagnosis-20260924.md`.
