# Complete raw graph callers for prepared convolution

Run `C:/Python313/python.exe -X utf8 -B` with `build.py`, then `audit.py` after
terminal execution. A failed numerical result is retained; no existing destination
is overwritten. This consumes the qualified normal Core `3c2f16b0` through its
existing test friend assembly, with no copied arithmetic implementation.

The explicit Probe.cs adaptation keeps all 312 geometries, 331 layouts, 2,648
bias/residual/ReLU cases, twenty supplemental cases and ten invalid/alias checks.
Actual product weight/input packing and raw arithmetic are checked. Its internal
nonfinite guard returns false without writing; the original component instead
performed its own public fallback. That explicit semantic change is tested.

Each of 2,668 cases additionally runs a budget-zero ordinary graph once and a
prepared graph twice: 2,668 control and 5,336 candidate requests. Graphs preserve
the existing ConvRelu/AddRelu behavior. Every non-NaN bit and NaN classification
must match; differing NaN payloads are recorded separately. Check inputs, held
owned results, exact rounded weights and actual requested scratch. Of these
cases, 2,504 use finite convolution and 164 use the original nonfinite fallback;
residual NaNs alone do not disable the preceding convolution.

Local qualification uses actual AVX2, CPU2 before startup, 12 GiB available
preflight, 8 GiB RSS, 900 seconds, 1 GiB available/output and 20 GiB disk bounds.
The consumer supports explicit AVX512 disabling for later AMD AVX2 qualification,
and actual AVX512 qualification. There is no performance measurement.
