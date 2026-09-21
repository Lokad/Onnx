# Profiler candidate: shared models and graph contracts

The unchanged, previously qualified `Replay.dll` runs against both frozen cores
from the profiler-allocation experiment. Eight local workers cover cache off/on,
baseline/candidate, and e5/shared-model modes. DINOv3, ResNet50 and GPT-2 retain
their complete native references and decoder state. E5 adds facade/context,
graph mutation, failure and ownership checks to the prior allocation proof.

Run `prepare.py`, `run.py`, and `audit.py` from the repository root using
`C:/Python313/python.exe -X utf8 -B`. Targets refuse overwrites. The artifact is
`artifacts/e5-profiler-shared-20260921`. No build, download, native inference,
production change or VM worker is introduced. All 664 arrays must pass native
`1e-4` and exact cross-core/cache comparisons before this local gate passes.
