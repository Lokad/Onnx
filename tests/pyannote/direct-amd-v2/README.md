# Direct-output convolution: prospective AMD application comparison

This successor preserves the first campaign's terminal-sample failure. All
nine setup/build commands exited zero; the last compiler sample had no live
members and failed the monitor before any model or timing run. Closure:
`5bfd1e51e4aa6788e9470be5b8be8d26db424af8cbbeb25fbb0ef8364ae9f8a4`.
The successor records a separate terminal transition only after the owned
root has exited and every recorded descendant is absent. Live samples keep
all original resource checks. Candidate bytes, workloads and gates are unchanged.

Compare selected production Core `e9c87932` / Data `85d166b5` with the isolated
direct-output candidate Core `19b9007d` / Data `cb6f86b0` and fresh Microsoft
ONNX Runtime 1.29.0. Pyannote is first, Parakeet second, Whisper deferred.
The earlier component gain is a reason to test this candidate, not a full
application speedup. All old samples and failed trials stay unchanged.

Before timing, build the candidate's normal source on Linux, prove all 3,113
Core and 697 Data methods/public declarations match the frozen runtime, and
run the complete backend/tensor suites (at least 3,311/343 passed). Execute
all 400 actual caller cases in normal and hardware-disabled modes. Retain
exact non-NaN bits and NaN classification; record all payload differences.
Run fresh complete Pyannote graphs/public requests and Parakeet native
qualification for both roles: 36 Pyannote arrays, 32 public requests and
1,568 Parakeet arrays. Require candidate Pyannote outputs and public results
bit-for-bit equal to production, plus all existing native bounds. The original
Parakeet arithmetic fix remains separate; no packing-budget change is included.

Then run native conformance and the candidate on ES2004a and IS1009a ten-minute
meetings plus 30-second recovery, preserving native timelines, centroid bounds
and ownership checks. Windows graph/shared/suite/package qualification is pinned.

Six fresh processes execute production, candidate, ORT, ORT, candidate,
production. The retained protocol calls the candidate role `portable` internally.
Each process has one warmup and three measured passes over the full 30-second
dialogue and three ten-second crops: 96 requests, including 24 warmups and 72
measured requests. Keep every sample. CPU2 affinity precedes runtime startup;
the monitor uses CPU0. ORT uses one intra/inter-op thread, sequential execution,
full optimizations and no spinning. No runtime/profile overrides during timing.

Preserve the existing integration gates: every role's full-dialogue process
mean max/min <=1.10, crops <=1.20; candidate full latency <=0.97 of production,
each crop <=1.05. The ultimate Lokad/ORT <=1.05 objective is unchanged. A failed
control or speed gate stays failed without an unchanged timing retry.

Limits: four hours per campaign, one hour per worker, 12 GiB owned RSS,
1 GiB available memory and tmpfs free, 2 GiB artifacts. Preflight requires
12 GiB memory and 3 GiB tmpfs. Track every process identity and native thread.
Use `/dev/shm/lokad-pyannote-direct-v2-20260922` because VM root disk is full.
The VM window is already exclusively authorized; no approval step remains.

Run from repository root using `C:/Python313/python.exe -X utf8 -B`:
`tests/pyannote/direct-amd-v2/prepare.py`, inspect successful preparation/selftests,
then `tests/pyannote/direct-amd-v2/finish.py` once. New artifacts are
`artifacts/pyannote-direct-amd-payload-v2-20260922` and
`artifacts/pyannote-direct-amd-execution-v2-20260922`. Collect only after terminal
owners, independently audit every gate, and preserve any failure. Product
promotion requires passing complete evidence and normal root/package checks.
