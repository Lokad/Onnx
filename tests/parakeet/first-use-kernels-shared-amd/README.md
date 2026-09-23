# M43 shared-model and e5 regression qualification

Run only after the complete M43 Parakeet comparison is admitted. Use the
unchanged Replay consumer and original native fixtures in four fresh processes:
selected-shared, selected-e5, candidate-shared, candidate-e5. Both roles cover
166 arrays and 5,000,814 values, including retained outputs, unchanged inputs,
context reuse, memory policy and five e5 sequence/padding cases.

Every candidate float output must be byte-identical to the fresh selected
release. Independently compare both roles against every original ORT output
with the original 1e-4 scaled error bound. No arithmetic exception or retained
result substitutes for a fresh process. This lane provides no performance score.

Use `C:/Python313/python.exe -X utf8 -B` followed by this directory's `run.py`
and `prepare`, `stage`, `launch`, `observe`, then `collect`; finish with
`audit.py`. Freeze all tools and fixtures before staging. Compute on AMD CPU 2
with CPU 0 monitoring, SDK/runtime unchanged. Preflight requires 12 GiB available
memory and 3 GiB tmpfs; each worker has 8 GiB RSS and 900 seconds. Preserve a
failure and its identities without relaunching or changing a gate.
