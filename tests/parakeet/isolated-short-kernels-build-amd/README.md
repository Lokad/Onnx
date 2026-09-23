# Isolated short-projection build and compiled scope

Four AMD jobs verify SDK10.0.204, restore/build the CLI normally, then inspect
every Core/Data method. Reuse the qualified flag-aware Bridge inspector.

All 3,179 existing Core and 697 Data methods keep exact bodies and flags except
four resolved call operands in existing matrix callers. Original MathOps kernels
and the general dispatcher stay intact. Add exactly two private Tensor helpers
and four internal arithmetic copies. Flags are 256/8 for the helpers and 512
for each arithmetic copy; no existing flags or public APIs change. Permit only
the explicitly proven static-initializer lambda rename, with identical body and
one resolved function-pointer operand. Compare new bodies against their exact
selected/M43 originals after declared callee substitutions only.

From root use `C:/Python313/python.exe -X utf8 -B` with this directory's `run.py`
commands prepare, stage, launch, observe, collect separately, then audit.py.
Freeze before execution; keep all failures and refuse existing outputs.
Build under source/global.json with --tl:off; CPU2 computes, CPU0 monitors.

Limits: 10 GiB available/3 GiB tmpfs preflight, 8 GiB owned RSS, 900 seconds/job,
1 GiB minimum available/tmpfs, 1 GiB output/job, 2 GiB campaign and four hours
total. Verify all thread affinities and monitoring gaps under ten seconds.
This lane provides no numerical, native-code or performance admission.
