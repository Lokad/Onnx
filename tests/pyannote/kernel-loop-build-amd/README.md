# Isolated fixed 3x3 kernel build

Build the M23 source prepared by `../kernel-loop-unroll`, using SDK 10.0.204
and the existing offline feed on the exclusively assigned AMD VM. The measured
M22 Core/Data pair is the control. No root product files change.

Run from the repository root with Python 3.13, `-X utf8 -B`:

1. `tests/pyannote/kernel-loop-build-amd/run.py prepare`
2. `tests/pyannote/kernel-loop-build-amd/run.py stage`
3. `tests/pyannote/kernel-loop-build-amd/run.py launch`
4. Observe with `run.py observe` until the original supervisor and its children
   are terminal. A timeout is not permission to relaunch.
5. `tests/pyannote/kernel-loop-build-amd/run.py collect`
6. `tests/pyannote/kernel-loop-build-amd/audit.py`

The fresh local artifact is `artifacts/pyannote-kernel-loop-build-amd-20260922`;
the remote directory is `/dev/shm/lokad-pyannote-kernel-loop-build-20260922`.
Existing destinations are refused. Preserve failed attempts.

Four jobs run in order: SDK identity, CLI restore, CLI Release build, complete
compiled-method inventory. Exactly Kernel512 may change, with 3,162 other Core
methods, all 697 Data methods and public declarations equal. No added or removed
methods and no compiler-name exceptions are accepted. CPU2 is set before worker
startup; the monitor uses CPU0. Limits are frozen in `protocol.py`.

This establishes a compiled candidate only. Full raw/wide/span/captured-layer
numerics under AMD AVX512 and AVX2, generated-code inspection and the fixed
complete-call performance screen are required before application qualification.
Neither this build nor its success is a speed or numerical correctness claim.
