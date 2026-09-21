The original report exporter wrote both complete report files, then its final
console print raised `NameError` because `json` was not imported. The measurement,
application audit, comparison audit and continuation closure had already passed
their execution checks; the timing verdict remains **failed controls**.

`report_v2.py` supplies that import. For the existing report it independently
checks the complete observations against the closed analysis, verifies all
closed file pins and checks every displayed timing/control value. It leaves
both original reports and the failed exporter unchanged. The receipt is
`artifacts/pyannote-vector-bias-report-20260921/checked.json`. No inference or
timing retry was performed.
