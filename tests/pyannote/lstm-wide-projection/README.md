# Isolated M25 wider LSTM projection

Current Pyannote attribution leaves8.1–8.5% of full-request sampled thread time
in the two explicit projection helpers. Actual selected AMD code uses eight
portable float lanes even when AVX512 is enabled. This prototype tests sixteen
independent outputs per vector while preserving each output's reduction order.
No speed gain, numerical qualification or root integration is claimed here.

From repository root run `C:/Python313/python.exe -X utf8 -B
tests/pyannote/lstm-wide-projection/prepare.py`. It verifies the closed current
profile and normal-root qualification, copies416 exact source files, changes
only panel dispatch, and adds two wide helpers plus a differential test file.
The exact diff and all hashes are retained in
`artifacts/pyannote-lstm-wide-projection-20260922`. Existing destinations are
refused. Root product files remain unchanged; no local build is performed.

Create enables wide dispatch only when UseIntrinsics, Vector512 acceleration
and AVX512F support all permit it. Single-row input/recurrent projection uses
four16-float accumulators. Four-row input projection uses two16-float vectors
per row. Both retain increasing reduction order and separate multiply/add;
portable vector loops and scalar tails retain their original boundaries.
The original two portable helpers remain byte-exact. No scratch, public API,
gate arithmetic, mutable-weight behavior or output ownership changes.

The new tests compare raw bits to those unchanged portable helpers across
zero reductions, output tile boundaries/tails, strided/reverse inputs,
one through four rows, nonfinite values/NaN payloads, signed zeros, sentinels
and unchanged operands. A complete bidirectional trajectory compares Auto,
Simd and Scalar and checks mutated weights and retained outputs. These tests
are prepared here and must run on the actual AMD modes before qualification.

Next build on AMD with SDK10.0.204/runtime10.0.8 and inspect the complete
compiled scope. Qualify all12 original captured LSTM graphs, all outputs and
native errors in AVX512-enabled/disabled and scalar modes; separately verify
SIMD-only dispatch. Inspect actual generated code before the complete-call
screen. Retain2352 timing clocks and the existing10% aggregate/5% per-node
component gates. Only an admitted candidate proceeds to complete application,
shared/e5, Parakeet, long-meeting, package and matched ORT qualification.

The full living implementation plan is
`.agent/m25-pyannote-wide-lstm-20260922.md`; preparation preserves a snapshot.
Pyannote remains first, Parakeet second and Whisper deferred.
