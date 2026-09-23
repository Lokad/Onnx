# Wide-entry shared graph regression comparison

Run only after the M54 complete Parakeet application, shared/e5 tensor checks
and complete Pyannote qualification pass. Compare the selected Core521bae17,
candidate Core672e5f30 and Microsoft ORT1.29.0 on the same AMD CPU2.

Reuse the qualified warmed ReleaseBenchmark.dll (d827e3b9) and its complete
method proof. No build jobs execute. The worker, native consumer, numerical
checks and statistics remain byte-identical to the original warmed lane.

All eight cases retain 600 fixed warmups and 180 measurements in each of six
fresh processes: current, candidate, ORT, ORT, candidate, current. Run all24
three-call numerical processes first. Keep all37512 calls,8640 measurements
and72 setup intervals. Every candidate output must match fresh selected bytes;
all roles retain fresh ORT scaled error<=1e-4 and ownership checks. All24
process controls must be<=1.10, every candidate/current ratio<=1.05.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`,
`launch`, `observe`, `collect`, then `audit.py`. Freeze tools before staging.
CPU0 monitors; preflight12GiB available/3GiB tmpfs, worker RSS8GiB,
900seconds/job and four hours/campaign. Preserve every failure and all clocks;
no unchanged scored retry, changed gate or favorable block selection.

The full prospective plan is copied into each prepared artifact bundle.
Root integration and BENCHMARK.md changes require further release checks.
