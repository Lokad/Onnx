# First-use matrix kernel build

M43 is an isolated Parakeet candidate: the M41 short-projection dispatcher plus
`AggressiveOptimization` on four existing float matrix methods. This lane builds
normally on the exclusive AMD VM with SDK 10.0.204 and an offline package feed.
It does not run inference, admit a performance result or change root product code.

From the repository root, use `C:/Python313/python.exe -X utf8 -B` with
`tests/parakeet/first-use-kernels-build-amd/run.py` and, in order, `prepare`,
`stage`, `launch`, periodic `observe`, then `collect` after all owners terminate.
Run `tests/parakeet/first-use-kernels-build-amd/audit.py` to close the result.
Existing artifact namespaces are refused. Keep failed outputs; never overwrite
or silently resume a worker.

Seven jobs check the SDK, restore/build the metadata inspector, restore/build
the CLI and product, then compare both selected release and M41 binaries with
the candidate. Resolved method bodies, exception regions, local declarations,
public interface and method implementation flags are retained. Against M41, all
3,181 Core and 697 Data method bodies must be identical, and only four float
methods may change implementation flags from 0 to 512. The double overload must
remain unchanged. Against current, the previously established wrapper and two
helpers are the only additional body changes. Compiler-name normalization is
restricted to the established static-initializer rename.

The seven jobs retain PID/birth identities, every resource observation, all
output and hashes. Compute uses CPU 2 and monitoring CPU 0. Fixed preflight and
worker bounds are in `protocol.py`; they must not be loosened to obtain a pass.
`test_checks.py` exercises rejection of collateral metadata changes.
