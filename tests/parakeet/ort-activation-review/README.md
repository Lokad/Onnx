# Identify the existing ORT activation path without another inference run

The exact-export graph has 72 QuickGelu nodes with alpha 1. Pinned source routes
these through MlasComputeSilu. The internal native leaf is now identified in
[the completed review](diagnosis-20260926.md), using the original samples.
Existing perf samples preserve every instruction address and request interval;
the two retained disassembly ranges cover GEMM and normalization only.

Copy the installed 30,112,440-byte ELF that those samples used, with its full
ff54b93f hash. Read it only while graph supervisor 1126961/birth1790456331.32
is between workers: every preceding job must be complete/code0, then suspend
that exact supervisor, confirm no children or other inference owners, stream
the immutable file, and resume the same supervisor in finally. The remote
operation has a 45-second alarm. The helper changes no remote file. It reports
ready=false without copying or pausing if a job is active; this is an observation,
not a retried inference experiment. Preserve any actual inspection failure.

The following capture already completed successfully. **Do not run it again.**

    C:/Python313/python.exe -X utf8 -B tests/parakeet/ort-activation-review/copy_binary.py

Only boundary observations may repeat. After one successful transfer, use the
local ELF, available objdump and retained samples/source for analysis. Do not
call this helper again, change the installed library, build an ORT replacement
or run inference. Do not infer a function name from the nearest exported symbol
in this stripped library. Check the applicable instruction sequence, constants,
call targets and actual sampled addresses before naming the path. This evidence
will guide the next explanation; it selects no polynomial or kernel variant.

Local inspection: artifacts/parakeet-ort-activation-review-20260926.
Original samples: artifacts/parakeet-ort-native-samples-20260924, kept unchanged.

The subsequent local analyzer checks the closed sample provenance, raw perf
addresses, original request windows, ELF mapping and build ID, exact pinned git
source, all 14 constants, instruction bytes, unwind boundaries and sampled
caller. It runs no inference and contacts no VM. Its default is read-only:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/ort-activation-review/analyze.py

The one-time `--publish` operation already closed at `41f851c4`. Preserve that
closure and all original reports. This is identification by source/instruction
correspondence plus runtime samples, not a source rebuild with identical bytes.
This diagnostic selects no new implementation or repeat of the rejected vector
screen; the plan first refreshes attribution after qualification.
