# Actual Pyannote convolution operands

Capture all 36 convolution call forms on each of the three retained embedding
inputs. Map native optimized nodes to original nodes by their unique unchanged
bias initializer. Follow only the native-declared residual and activation chain;
downsample fallbacks retain their raw convolution endpoint.

Adding graph outputs can change optimization. These are component fixtures, not
timing measurements. Preserve all original nodes, initializers and the main
output. Compare that output with a fresh unmodified native session and the
retained baseline at the unchanged `1e-4` scaled bound. Require exact repeats,
unchanged inputs and held outputs. Store each distinct tensor only once.

From repository root, using `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/blocked-spatial-fixtures/run.py
    tests/pyannote/blocked-spatial-fixtures/audit.py

The new artifact is `artifacts/pyannote-blocked-spatial-fixtures-20260922`.
Refuse an existing directory. Pin source, model, packages and prior evidence;
declare exact tensor storage before inference. CPU2 before native initialization,
12 GiB preflight, 8 GiB RSS, 900 seconds, 1 GiB output and 20 GiB free disk.
Collect all 108 calls, including twelve fallback calls across three inputs.
No root product mutation or model/dependency download is required.
