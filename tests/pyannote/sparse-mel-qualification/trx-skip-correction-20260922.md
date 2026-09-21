# Correction: the backend report retained 93 skipped tests

The September21 sparse-mel application report's backend skipped-test column
incorrectly says zero. The correct result is **3,280 passed, 93 skipped, zero
failed**. Request-focused remains2passed/0skipped and tensors342passed/0skipped.

The original TRX has3,373individual outcomes:3,280Passed and93NotExecuted.
Its aggregate total/executed/passed counters are3373/3280/3280, but the aggregate
notExecuted field is0. The exporter used that aggregate field for the skipped
column. The independent application auditor already checked all individual
outcomes and required93skips; its qualification verdict is unchanged.

The original report, exporter, TRX and closed evidence remain preserved. This
correction changes no tests, numerical bound, public result, timing sample or
benchmark ratio. Future exporters count skipped individual records and retain
the raw aggregate counters separately.

Evidence is artifacts/pyannote-trx-skips-20260922/closed.json. Run
verify_skip_counts.py using C:/Python313/python.exe -X utf8 -B to reproduce in
a fresh output location; the existing correction is immutable.
