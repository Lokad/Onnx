# Build and qualify the direct depthwise candidate on AMD

This uses the established serial resource supervisor, offline package feed and
SDK 10.0.204. CPU 2 runs every worker and thread; CPU 0 monitors. Build jobs need
2 GiB available memory and 1 GiB tmpfs, and are bounded at 3 GiB RSS/180 seconds.
Each focused test process has the same memory bounds and a 300-second timeout.
At least 1 GiB memory/tmpfs stays free; total staged/output storage stays below
512 MiB. No model inference or performance scoring happens in this workflow.

Compile scope: one changed original Core method, four added private helpers,
3,276 unchanged Core methods, all 697 Data methods unchanged, all public surfaces
and existing implementation flags unchanged. Data is copied from the original
qualified runtime byte-for-byte. Only the original two nullable warnings,
repeated in the build summary, are accepted.

The eight focused tests run normally and with hardware intrinsics disabled.
They load the exact original M78 Core separately to compare actual outputs.
All 59 observed geometries must match bit-for-bit; NaN classification is checked
for explicit special-value cases. Every input remains unchanged and public
outputs stay independent. Complete per-geometry reports and test names are kept.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root:

    tests/parakeet/direct-depthwise-build-v2/run.py prepare
    tests/parakeet/direct-depthwise-build-v2/run.py stage
    tests/parakeet/direct-depthwise-build-v2/run.py launch build
    tests/parakeet/direct-depthwise-build-v2/run.py observe build
    tests/parakeet/direct-depthwise-build-v2/run.py collect build
    tests/parakeet/direct-depthwise-build-v2/review.py build
    tests/parakeet/direct-depthwise-build-v2/run.py launch capture
    tests/parakeet/direct-depthwise-build-v2/run.py observe capture
    tests/parakeet/direct-depthwise-build-v2/run.py collect capture
    tests/parakeet/direct-depthwise-build-v2/review.py capture

Observe actual deployed owners to terminal status before collecting. Every
operation except observation is one-time. Preserve a failed build/test/reviewer;
an analysis failure does not authorize repeating inference or changing a gate.
Focused contracts do not qualify native model correctness, mechanism counts,
performance, shared-model regressions or release admission.
