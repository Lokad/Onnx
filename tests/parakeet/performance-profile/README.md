# Complete Parakeet wall attribution

This diagnostic lane profiles the retained twenty-clip Parakeet corpus using
unchanged production Core/Data assemblies. It does not run Microsoft ORT or
modify the pending AMD comparison. Pyannote remains the first optimization
priority; this prepares Parakeet's next measured target while e5 owns the VM.

The trace worker runs one complete unprofiled trajectory pass and one wall-profiled
pass. A separate fresh worker runs twenty public `Transcribe` controls. Both traced passes call
the existing managed generation method and carry their own encoder outputs and
accepted recurrent states. They capture all 1,200 decoder calls per pass:
2,480 graph calls, 12,160 input arrays and 9,760 output arrays altogether.
The second pass must retain all input/output bits. Every public result must
match the original native transcript, tokens, frames and duration decisions.

Graph timers cover `GraphExecution.Execute`. Context reset has a separate clock.
Tensor serialization, hashing and ownership checks are outside graph timers.
The public controls time the complete API without diagnostic callbacks. These
different instruments cannot be subtracted to estimate API overhead, or combined
with historical native timings to claim a new speedup or parity result.

From the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/performance-profile/prepare.py
    C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/parakeet/performance-profile -p test_*.py -v
    C:/Python313/python.exe -X utf8 -B tests/parakeet/performance-profile/run.py trace
    C:/Python313/python.exe -X utf8 -B tests/parakeet/performance-profile/run.py public
    C:/Python313/python.exe -X utf8 -B tests/parakeet/performance-profile/audit.py

Preparation copies source into `artifacts/parakeet-performance-profile-v2-20260921`
and builds against the exact frozen product DLLs. It validates each retained
model/PCM digest and the original native public-reference provenance. Restore
uses an existing offline NuGet feed. Commands refuse to overwrite their
artifacts; inspect actual process identity and state before any continuation.

The worker uses normal .NET settings on logical CPU 2 inherited before runtime
startup. The supervisor uses CPU 0. Preflight waits at most 15 minutes for
10 GiB available memory. Execution permits 1,800 seconds, less than 8 GiB RSS,
at least 1 GiB available memory and 20 GiB free disk, and at most 1 GiB output.
Failures retain partial captures and logs, and only the owned worker may be
terminated. Unrelated workstation processes and VM campaigns remain untouched.

The independent auditor reads every captured tensor, reconstructs every greedy
decision and checks blank-state retention, carried states, selected encoder
frames, complete node coverage, raw clocks and resource observations. Its tests
deliberately corrupt clocks and decoder trajectories. Existing Parakeet native
numerical discrepancies remain separate; this lane proves profiling neutrality
and public agreement, not a new full tensor-native conformance result.

The first combined worker completed both graph passes but exceeded the unchanged
8 GiB RSS limit during its public phase. Its complete captures and failure
receipt remain in the unsuffixed artifact; it has no success verdict. The
successor reads ordinary dense tensor spans directly when hashing/writing them,
avoiding diagnostic copies, and separates public controls from capture memory.
Seven layout self-tests verify byte order, including reversed and broadcast
views; a nonfinite-value test verifies refusal. No inference arithmetic or runtime
numerical setting changes, and no forced collection is added.
