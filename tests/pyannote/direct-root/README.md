# Conditional root integration of direct-output convolution

This step requires the completed direct-amd-v2 campaign, terminal owners and
an independently audited passing performance verdict. It refuses to change
source before all original correctness, meeting, repeatability and speed
gates pass. A component pass or an incomplete campaign is insufficient.

From repository root, run `C:/Python313/python.exe -X utf8 -B` with
`tests/pyannote/direct-root/run.py`, followed by `audit.py`. The tools use
`artifacts/pyannote-direct-root-20260922` and refuse existing output.

Only the reviewed four-file patch is applied: the tiled convolution caller,
the checked direct-output wrapper, its generated kernel and focused tests.
Normal CLI/backend/tensor project references remain. The build must match all
3,113 Core and 697 Data methods and public declarations of the measured
candidate. The full Windows backend suite must pass 3,311 tests with 93
existing skips; the tensor suite must pass all 343 tests.

The NuGet package must contain the actual root-built Core and retain its
Google.Protobuf 3.33.5 dependency. An independent PackageReference consumer
checks public operators, model import, input/held-output ownership and every
value of the admitted tiled-convolution route. The final audit snapshots the
editable source and verifies process termination and resource limits.

Build preflight requires 8 GiB available memory; tests/consumer require 10 GiB.
Each owned tree stays below 8 GiB RSS, 900 seconds and 1 GiB output, with at
least 20 GiB free disk. All .NET commands disable the terminal logger. The
tools preserve any failure and never retry an existing stage automatically.
They do not publish a package or push commits. Timing belongs to the measured
candidate; method equivalence does not create a separate root-rebuild score.
