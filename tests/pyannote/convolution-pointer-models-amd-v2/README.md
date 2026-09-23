# Complete M28 Pyannote output qualification

The closed full-product campaign `4727eadc` qualifies the admitted M28
candidate, Core `e776cec2` / Data `0c55b650`. This campaign compares it with
the current selected Core `208371f6` / Data `b9358370`, using fresh AMD
execution of both complete models and public diarization requests.

Run from the repository root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/convolution-pointer-models-amd-v2/run.py prepare
    tests/pyannote/convolution-pointer-models-amd-v2/run.py stage
    tests/pyannote/convolution-pointer-models-amd-v2/run.py launch
    tests/pyannote/convolution-pointer-models-amd-v2/run.py observe

Only collect after terminal state, then run `audit.py`. Never relaunch or
overwrite a failed attempt; do not observe a closed campaign.

Five jobs build the candidate consumer, verify that its only compiled change
is the expected Data identity literal (95 of 96 methods unchanged), then run
selected and candidate complete qualification. All 18 graph-output arrays,
2,917,107 values and 16 public requests per product must pass the original
native error limits. Candidate arrays and complete public results must match
fresh selected outputs exactly. Repeatability, read-only inputs and ownership
checks remain unchanged. The preceding three checker tests are retained in
the original M22 campaign; the complete checkers and native auditor are reused
byte for byte. The selected consumer is the closed M22 consumer, rather than
the preceding M17 consumer. Model/audio/reference assets remain unchanged.

Pin global.json to SDK 10.0.204; workers use runtime 10.0.8 and CPU2. Keep the
original 12 GiB available/3 GiB tmpfs preflight, 8 GiB owned-RSS cap, 1 GiB
live available/tmpfs floor, 900-second job and 2 GiB artifact limits. This is
numerical qualification; its diagnostic clocks are not a performance score.
The selected root and application/ORT results remain unchanged.

The original local preparation stopped before staging because text-mode copying
normalized the frozen auditor's line endings. Failure closure `01baff87` is
retained and required. This successor copies both original auditor files byte
for byte. No product, comparator or numerical bound changed.
