# One complete M78 depthwise count observation

Build the frozen `depthwise-route-source` observer on AMD. The compiled inventory
must retain all 3,277 original Core methods with changes only in the three planned
methods; all 697 Data methods and all public surfaces/implementation flags stay
unchanged. Only the 15 private observer methods may be added. The exact original
Data and common consumer binaries are reused. The source insertions reverse
exactly, preserving arithmetic and dispatch. There are no added warnings.

The counter capture runs the unchanged complete public Parakeet protocol once:
20 corpus warmups plus 60 measured requests, retaining every record and checking
exact transcripts, tokens and decoder calls. These elapsed times are not scored.
Counter coverage is all four corpus passes, with all 59 distinct depthwise
geometries reconciled against the closed source/shape diagnosis. Every partial
panel, product shape, actual leaf, input/dense layout and scratch bucket is kept.
All per-shape counts must match; aggregate agreement alone is insufficient.

The VM remains serial. Workers and every child thread stay on CPU 2; the supervisor
uses CPU 0. Builds require 2 GiB available memory / 1 GiB tmpfs and stay below
3 GiB RSS for at most 180 seconds per job. Capture requires 11 GiB available memory
/ 2 GiB tmpfs and stays below 12 GiB RSS for at most 900 seconds. Every job leaves
at least 1 GiB memory/tmpfs free; stage and output total stays below 512 MiB.
Half-second samples and original before/after CPU accounting are retained, with
the existing 1% foreign-CPU bound and its short-lived-process limitation.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root:

    -m unittest discover -s tests/parakeet/depthwise-route-amd -v
    tests/parakeet/depthwise-route-amd/run.py prepare
    tests/parakeet/depthwise-route-amd/run.py stage
    tests/parakeet/depthwise-route-amd/run.py launch build
    tests/parakeet/depthwise-route-amd/run.py observe build
    tests/parakeet/depthwise-route-amd/run.py collect build
    tests/parakeet/depthwise-route-amd/audit.py build
    tests/parakeet/depthwise-route-amd/run.py launch capture
    tests/parakeet/depthwise-route-amd/run.py observe capture
    tests/parakeet/depthwise-route-amd/run.py collect capture
    tests/parakeet/depthwise-route-amd/audit.py capture

Observe the actual deployed owner to terminal status before collecting. All actions
except observation are one-time. Preserve every failed state; do not repeat a
workload after a collector or reviewer failure. No optimization or release
promotion is authorized by a successful count capture. Its result answers which
mechanism a subsequent single candidate must address.
