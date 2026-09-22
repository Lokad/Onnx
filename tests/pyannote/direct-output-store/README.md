# Ordinary stores for narrow direct-output tails

The prior direct-output screen improves its equal-shape mean 6.73%, but two
two-column geometries regress 12.69% and 5.35%; it remains rejected. This
successor changes the store mechanism, retaining every shape and every gate.

Only the three final masked stores and their unused integer casts change.
The new private helper writes exactly one-to-seven floats using ordinary
16-byte, 8-byte and scalar stores. Input masking, ascending arithmetic,
explicit bias NaN selection, destination stride, admission and the original
two-row remainder stay unchanged. No workload is excluded.

The exact v4 consumer and shape manifest are retained: 2,882 cases in each of
normal/forced-scalar Windows/AMD modes, all 22 real timing geometries and 33
checked actual offsets. Timing includes clearing, packing, multiplication and
bias/copy, with preallocated buffers. The four-process order and six measured
blocks per shape are unchanged. Fixed gates remain process ratio <=1.10,
geometric mean <=0.95 and every shape <=1.05. Passing permits complete product
and model qualification only, not an application or ORT speed claim.

Prefix these commands with `C:/Python313/python.exe -X utf8 -B` from the root:

    tests/pyannote/direct-output-store/run.py prepare
    tests/pyannote/direct-output-store/run.py stage
    tests/pyannote/direct-output-store/run.py launch
    tests/pyannote/direct-output-store/run.py observe
    tests/pyannote/direct-output-store/run.py collect
    tests/pyannote/direct-output-store/run.py audit

The wrapper verifies the previous closure and preserves its failed gate. New
artifacts use `artifacts/pyannote-direct-output-store-20260922` and the matching
`/dev/shm/lokad-pyannote-direct-output-store-20260922` VM directory. All original
process, runtime, affinity, resource, ownership and no-overwrite checks apply.
Observe one owner until terminal; do not repeat a completed launch or timing
screen. Production and accepted application benchmark tables remain unchanged.
