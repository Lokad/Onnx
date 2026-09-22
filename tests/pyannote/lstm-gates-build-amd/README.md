# Build and constrain the isolated default-gate LSTM product

From repository root with C:/Python313/python.exe -X utf8 -B run test_checks.py,
then run.py prepare,stage,launch,observe,collect and audit.py. Destinations are
artifacts/pyannote-lstm-gates-build-amd-20260923 and
/dev/shm/lokad-pyannote-lstm-gates-build-20260923. Existing destinations fail.

Source receiptd836e93c0eae60b5c8729685d2097b0257320cdac1f1c536777938dbcbd5ef63
pins all418 source files, including416 qualified current files (one recurrence
dispatch edit) and two additions. Copy all selected runtime dependencies,
not just Core/Data. Preserve the repository global.json selectingSDK10.0.204.
Use the previously verified Bridge binary and complete instruction inventory.
Only Lstm may change; add exactly one private
LstmUpdateDefaultGates helper. Every other existing Core method and all697 Data methods
must be unchanged; public declarations must match. No model inference occurs.

Four jobs identify SDK10.0.204, restore offline, build the normal Release CLI
and inspect compiled scope under runtime10.0.8. Use --tl:off and no shared build
servers. Worker CPU2/monitor CPU0;12GiB available/3GiB tmpfs preflight,8GiB owned
RSS,900seconds per worker,1GiB minimum available/tmpfs/output,2GiB artifacts.
Retain every process/thread identity and resource observation. Collect only
terminal owners; independently replay all checks before closing. No numerical,
speed or root-integration claim follows from build success alone.
