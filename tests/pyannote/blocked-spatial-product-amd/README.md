# Actual prepared-convolution product on AMD

This is numerical and ownership qualification, with no speed measurement.
Run `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `run.py stage`,
`run.py launch`, `run.py observe`, then `run.py collect` only after all owners are
terminal; finally run `audit.py`. Existing destinations are refused. Observe the
same PID/birth after any interruption; do not relaunch because a poll expired.

Preparation rebuilds the already-qualified LayerGraphs consumer with only its
expected Core hash changed to the corrected normal product `3c2f16b0`, then checks
all local captured layers again. It uses the exact qualified RawGraphs executable
and the existing 509,207,936-byte fixture set. Both product consumers, sources,
local outputs, remote tools and external/runtime assets are pinned before transfer.
Remote monitor/transport changes are retained as diffs from executed predecessors.

Exactly four fresh workers run: raw-256, raw-512, layers-256, layers-512. AVX2 jobs
explicitly set `DOTNET_EnableAVX512F=0`; AVX512 jobs use the ordinary environment.
Actual product lane selection must match the job label. Each raw mode preserves
all 2,648 cases, twenty supplemental, ten invalid/alias checks and all 2,668 graph
controls / 5,336 candidate requests. Every layer mode preserves all 108 cases,
216 graph calls and 119,823,360 values, including native <=1e-4 checks.

CPU2 is inherited before runtime startup; monitor CPU0. All process and thread
affinities are retained. Require 12 GiB available/3 GiB tmpfs preflight, 8 GiB RSS,
1 GiB minimum available/tmpfs, 1 GiB output, 2 GiB artifacts, 900 seconds per worker
and four hours per campaign. Preserve the authorized VM service masks and account
lingering. This does not change full-application selection or the ORT baseline.
