# Longer-projection regression in the isolated candidate

M50 remains rejected by [the complete screen](screen-20260923.md). Its single
failing case is the captured matrix with 167 rows, reduction 1,024 and 4,096
columns: 16.939432 ms selected versus 19.041327 ms candidate, a 12.4083%
regression. All 48 repeatability controls pass. The two neighboring 167-row
geometries pass their regression gates.

The [fixed 20-call blocks](screen-blocks-20260923.csv) describe every clock in
all four processes and all 21 fixtures: 504 blocks, 10,080 calls. They are a
post-result diagnostic and change no score, inclusion rule or acceptance gate.
The failing fixture's measured blocks are:

| Process | Calls 60–79 ms | Calls 80–99 ms | Calls 100–119 ms |
|---|---:|---:|---:|
| Selected 0 | 16.911480 | 16.948436 | 17.000509 |
| Candidate 1 | 20.045291 | 19.225291 | 18.000861 |
| Candidate 2 | 19.831225 | 19.142786 | 18.002506 |
| Selected 3 | 16.992298 | 16.878361 | 16.905507 |

Both candidate processes show the same elevated interval and remain slower in
their final block. The clocks do not identify whether compilation, collection,
code layout or another cause explains the difference. They do not justify
discarding early measurements or increasing warmups retrospectively.

The selected and candidate arithmetic have exact IL bodies; their five common
optimized native methods match instruction text after the declared address and
branch-offset normalization. However, the candidate's short calls bypass the
original general dispatcher, changing its invocation history. The new dispatcher
also inlines the general path into a 4,092-byte method. The existing untimed code
capture uses 80 calls per fixture and does not trace the four scored processes
or capture all public callers, so it cannot establish which code ran during the
failed interval.

The next useful experiment is an untimed runtime diagnostic over the same
21-fixture, 120-call order with the actual selected and M50 DLLs. Reuse the
existing EventPipe collector/exporter, instrument explicit call boundaries,
retain all compiler and GC events, and independently capture every changed
public caller and both dispatch paths. Event method loads show availability,
not proof that a particular call executed that version. Keep the failed screen
closed; any subsequent scored candidate must be a distinct source change with
fresh qualification.
