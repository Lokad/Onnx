# M28 full product and package qualification

Run from the repository root using `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/convolution-pointer-product-amd-v2/run.py prepare
    tests/pyannote/convolution-pointer-product-amd-v2/run.py stage
    tests/pyannote/convolution-pointer-product-amd-v2/run.py launch
    tests/pyannote/convolution-pointer-product-amd-v2/run.py observe

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

Artifacts: `artifacts/pyannote-convolution-pointer-product-amd-v2-20260923`.
VM directory: `/dev/shm/lokad-pyannote-convolution-pointer-product-v2-20260923`.
The selected root and application/ORT scoreboard remain unchanged.

The original attempt stopped during staging before any worker launched:
the component screen's runtime omitted Data. Its failure closure `cb64be90`
is retained and checked. This successor copies the complete qualified build
runtime, independently checks both product hashes and its collection receipt,
and uses a fresh namespace. Product source and acceptance criteria are unchanged.
