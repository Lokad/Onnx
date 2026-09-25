# Observe the complete public candidate's depthwise work once

Build the frozen observer on AMD and reuse the exact qualified Data and common
public consumer. Compared with the candidate, only the three observer callsites
may change: 3,278 original Core methods and all 697 Data methods remain unchanged.
Only 16 private observer methods may be added. Preserve all public surfaces,
existing method flags and original warnings. The four new product helpers are
unchanged by instrumentation.

Run the unchanged full 80-request public Parakeet protocol once: 20 warmups and
60 measured requests. Compare transcripts, tokens and decoder calls exactly with
the original M78 results. Times are unscored. Across all four corpus passes,
require 2,080 direct operator completions over all 59 observed geometries, with
zero patches, tiled panels, temporary group views and generic matrix calls.
An aggregate match does not substitute for per-geometry reconciliation.

The VM remains serial. CPU 2 runs every worker/thread; CPU 0 monitors. Build
requires 2 GiB available/1 GiB tmpfs, with 3 GiB RSS and 180 seconds per job.
Capture requires 11 GiB available/2 GiB tmpfs, with 12 GiB RSS and 900 seconds.
Leave at least 1 GiB memory/tmpfs free; stage and output stay under 512 MiB.
Retain half-second samples and the original 1% foreign-CPU accounting limit.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root:

    -m unittest discover -s tests/parakeet/direct-depthwise-observer-amd -v
    tests/parakeet/direct-depthwise-observer-amd/run.py prepare
    tests/parakeet/direct-depthwise-observer-amd/run.py stage
    tests/parakeet/direct-depthwise-observer-amd/run.py launch build
    tests/parakeet/direct-depthwise-observer-amd/run.py observe build
    tests/parakeet/direct-depthwise-observer-amd/run.py collect build
    tests/parakeet/direct-depthwise-observer-amd/audit.py build
    tests/parakeet/direct-depthwise-observer-amd/run.py launch capture
    tests/parakeet/direct-depthwise-observer-amd/run.py observe capture
    tests/parakeet/direct-depthwise-observer-amd/run.py collect capture
    tests/parakeet/direct-depthwise-observer-amd/audit.py capture

Observe actual owners to terminal status before collection. Every operation
except observation is one-time. Preserve failures. A counter capture supplies
mechanism evidence, not a performance score or release admission.

The observer build reuses the retained qualified build package cache, with the
same offline feed, to avoid another duplicate cache in tmpfs.
