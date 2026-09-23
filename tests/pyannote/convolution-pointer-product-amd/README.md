# M28 full product and package qualification

Run from the repository root using `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/convolution-pointer-product-amd/run.py prepare
    tests/pyannote/convolution-pointer-product-amd/run.py stage
    tests/pyannote/convolution-pointer-product-amd/run.py launch
    tests/pyannote/convolution-pointer-product-amd/run.py observe

Observe only while active; collect once terminal, then run `audit.py`. Do not
relaunch, overwrite a failed attempt, or observe a closed campaign.

The admitted component closure is `ead36d62`. All 416 source files come from
the qualified isolated M28 tree. The normal SDK 10.0.204 build must match all
3,163 Core and 697 Data methods of the measured candidate. Both public
surfaces must match. Full suites must preserve the selected test census:
3,432 backend passes, 41 existing AMD skips, and 343 tensor passes. The
unchanged package consumer must import MNIST, preserve owned/read-only
values, and execute its two prepared graph calls against the actual nupkg.

Fourteen jobs run sequentially on CPU2 with CPU0 monitoring. This campaign
does not measure performance. Its prospective available-memory preflight is
10 GiB, providing 2 GiB beyond the unchanged 8 GiB owned-RSS ceiling. Live
available memory remains at least 1 GiB; tmpfs, output and elapsed limits
are unchanged. This avoids pausing ordinary builds solely because retained
historical evidence consumes tmpfs. All frozen numerical and performance
campaigns retain their existing resource rules.

Artifacts: `artifacts/pyannote-convolution-pointer-product-amd-20260923`.
VM directory: `/dev/shm/lokad-pyannote-convolution-pointer-product-20260923`.
The selected root and application/ORT scoreboard remain unchanged.
