# Actual Parakeet decoder LSTM fixtures

This prospective correctness lane retains both complete recurrent calls from all
190 selected decoder steps: English 37, French 29, JFK 46, token-limit 4,
frame-limit 37 and English repeat 37. Silence has no decoder call. The selected
native result and its original reference manifest are pinned from M63 models
closure `2d6aedee`. M63 application closure `5a7344ce` rejected that unrelated
candidate; current Core `672e5f30` and Data `065b7a7f` remain unchanged.

Each nonempty case runs four complete decoder trajectories with fresh execution
contexts and independently carried states: ordinary twice, capture twice. The
saved selected encoder output supplies the same actual frames. Every step must
match all four original output arrays exactly and keep the original token and
duration decision. Blank tokens retain incoming recurrent states. Holding extra
per-execution output bindings retains LSTM operands and endpoints; no ONNX input,
node, shape or graph output is edited. Every optimized node field/attribute and
all original initializer references, shapes and contents must remain unchanged.
Held output tensors survive later calls and resets unchanged. Expect 760 decoder
executions, 3,040 original array checks and 380 complete LSTM fixtures.

Two tiny ONNX files contain byte-exact copies of the original individual LSTM
nodes, original opsets/IR, and the observed fixed float shapes. W/R/B are fed from
the captured actual constants; models contain no duplicate weights. These
derived models are correctness instruments, not performance baselines. Native
ORT 1.29.0 runs each complete call twice with one intra/inter thread, sequential
execution, all graph optimizations, no spinning and CPU provider only. Require
all 2,280 output arrays to remain finite, have exact shapes, meet scaled absolute
error `abs(managed-native)/max(1,abs(native)) <= 1e-4`, repeat exactly, preserve
inputs, and remain independently owned after later calls. Retain every native
array and its worst error; no inference occurs on Windows.

The VM is exclusively authorized. Use new namespace
`/dev/shm/lokad-parakeet-decoder-lstm-capture-20260924`; workers inherit CPU2 and
the monitor uses CPU0. Five jobs run in order: SDK version, helper restore,
helper build, capture, native. Canonical global.json must select SDK10.0.204 and
runtime10.0.8. Only the helper is built; products and canonical models are reused.
No model download, new package or runtime switch is allowed. Build with
`--tl:off --nologo -v minimal`; use the existing offline package feed.

This decoder-only lane has prospective limits independent of the unchanged
full-model/application limits: preflight available memory 8 GiB and tmpfs 2 GiB;
owned RSS below 4 GiB; at least 1 GiB available memory and free tmpfs throughout;
128 MiB output per job, 512 MiB total campaign artifacts, 900 seconds per job and
four hours total. Retained unique capture tensors are additionally limited to
64 MiB. No encoder execution or 2.4 GB model copy is needed. Preserve all failures
and exact PID/birth ownership. Do not observe a closed campaign or retry failed
inference unchanged. Collection dereferences hardlinks into ordinary members.

From the repository root use `C:/Python313/python.exe -X utf8 -B` with
`tests/parakeet/decoder-lstm-capture-amd/run.py prepare`, then `stage`, `launch`,
`observe` while active and `collect` only after all owners terminate. Run
`tests/parakeet/decoder-lstm-capture-amd/audit.py` to independently reconcile all
fixtures, original output controls, native outputs and resource samples.

This lane establishes actual test inputs for the prepared-recurrence prototype.
It makes no optimization or speed claim. Candidate implementation, both
instruction modes, lifecycle/budget tests, frozen complete-call timings and the
full application admission remain required before any release change.
