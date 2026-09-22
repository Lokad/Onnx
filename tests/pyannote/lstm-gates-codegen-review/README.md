# M26 generated-code review

The frozen original auditor is in ../lstm-gates-codegen-amd. Its first local
audit rejected the interleaved candidate-256 gate body; all VM workers had
already succeeded. audit_v2.py retains that failed audit and all raw fragments,
then proves the bounded670-byte helper prefix exactly equals a clean optimized
body from candidate-512. It does not rerun any worker or change the original
auditor. review.py independently checks the admitted helper in all four modes.

Historical results are closed. Do not rerun audit_v2.py against the closed
destination or call the original run.py observe after closure. For inspection,
read the retained evidence and [report](../lstm-gates-results/codegen-20260923.md).
