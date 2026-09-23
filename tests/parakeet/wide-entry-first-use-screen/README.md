# Isolated Parakeet wide projections: complete-call comparison

M54 isolates the wide matrix entry and compiles it and its private packing
helper with full optimization on first use. Original entry/shared bodies and
flags remain exact. Numerical closure and full native review are required.
The consumer, fixture checker, scoring code, tests and bounds are byte-identical
to the rejected M41 screen. Preparation requires the new candidate's normal
build, numerical closure and complete generated-code review first.

Four fresh CPU-2 processes run current/candidate/candidate/current. Each visits
21 fixed matrix pairs with 60 warmups and 60 measured calls per case. All 10,080
clocks are retained; 5,040 are scored. The public destination MatMul boundary
includes validation, clearing, scratch rental/return, packing, consumption and
tails. Array IO and output validation stay outside the clock.

The first twelve fixtures retain the original capture order. Nine derived row
prefixes follow: M48 from the original M51 operands, then M61/M63 from M106,
each for the same three weight geometries. These are matrix fixtures, not new
recorded clips. Every prefix is checked against its immutable source bytes.

Require at least 10% improvement over the twelve cases at M48–63 and no case
over 5% slower. Same-role aggregate max/min is at most 1.10 and per-case max/min
at most 1.20. Candidate process aggregates must be strictly below both current
aggregates for both the target twelve and all 21 cases. The unchanged nine-case
aggregate is also published and checked for repeatability. The historical
"unchanged" label in the scoring schema means the nine original longer-row
comparison cases; The isolated candidate routes those through the isolated copies too. Their
original regression and stability gates remain. All thresholds use
exact rational clocks. No favorable rerun or threshold relaxation.

Run `C:/Python313/python.exe -X utf8 -B` with `test_score.py`, then `run.py prepare`,
`stage`, `launch`, `observe`, `collect`, and `audit.py`. An admitted component
still needs complete native/public/shared/suite/package and application checks.
Transport omits existing fixtures; verified hardlinks reuse the terminal numerical
lane's arrays. No models are copied.
