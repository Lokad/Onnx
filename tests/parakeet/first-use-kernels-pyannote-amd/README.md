# M43 complete Pyannote regression qualification

After the full M43 Parakeet comparison is admitted, build the existing graph
consumer with exactly one changed Data-assembly identity string. Its inventory
must show 95 unchanged methods and one method differing by that literal only.
No test arithmetic or acceptance bound changes.

Run fresh selected and candidate processes. Each covers 18 complete native graph
outputs (2,917,107 values) and 16 public diarization calls. All candidate outputs
and public results, including centroid values, must exactly match the selected
release. Both roles independently retain the original ORT scaled-error bound
of 1e-4, input and held-output checks and public expectations. Diagnostic clocks
and node profiles are retained; this lane does not measure a performance gain.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, then `audit.py`. Freeze before any VM execution. Runtime
and SDK remain 10.0.8 and 10.0.204. CPU 2 computes and CPU 0 monitors; preflight
requires 12 GiB available and 3 GiB tmpfs, with 8 GiB RSS and 900 seconds per
worker. Reuse immutable inputs through hardlinks; unlink targets before replacing
an assembly. Retain failures without changing acceptance or repeating inference.
