# Normal AMD build for Winograd input reuse

Build the isolated M36 source against current measured Core521bae17/Dataf3b9aa81.
All420selected root files remain unchanged. The complete compiled inventory
must contain exactly one changed private method, MultiplyWinograd512; all
3,178other Core methods,697Data methods and public declarations remain exact.
No added/removed methods or compiler-renaming exceptions are allowed.

Four jobs use SDK10.0.204/runtime10.0.8 and the pinned offline feed: SDK check,
CLI restore, normal Release build and complete method inventory. Workers CPU2,
monitor CPU0. Bounds:10GiB available/3GiB tmpfs preflight,8GiB owned RSS,
900seconds per job,1GiB minimum available/tmpfs/output and2GiB artifacts.
This is build qualification only, with no model execution or timing claim.

From root use C:/Python313/python.exe -X utf8 -B with test_checks.py, then
run.py prepare/stage/launch/observe/collect and audit.py. Collect only terminal
owners. Refuse existing artifacts/pyannote-winograd-output-blocks-build-amd-20260923
and /dev/shm/lokad-pyannote-winograd-output-blocks-build-20260923. Preserve all
failed evidence and immutable selected runtimes; do not rerun old campaigns.
