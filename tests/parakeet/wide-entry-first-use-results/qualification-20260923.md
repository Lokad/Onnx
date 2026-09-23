# Wide entry numerical and native qualification

All four numerical workers pass: selected/candidate with normal instructions
and AVX512disabled. Each covers68groups and7,749,681initial values, with complete
finite-bit/nonfinite checks and1,560independent coordinate checks. The original
66groups remain unchanged. Two added Parallel(2) cases exercise the cloned
callback at64and106rows, both with1024reduction/columns.

Both parallel cases preserve destination identity, poisoned repeated outputs,
retained allocating output, input arrays and surrounding guards. The64row case
splits32/32 with zero scratch. The106row case splits53/53: selected scratch is
zero and candidate scratch is16,777,216bytes across two destination calls,
matching four packs and returns. Every semantic observation agrees across
products and instruction widths.

The51complete emitted native bodies are retained and every branch label
resolves. The new entry is5,625bytes of first-use FullOpts; its packing helper
is567bytes of first-use FullOpts. All four arithmetic functions remain separate
calls and their normalized instruction streams exactly match the reviewed M52
copies: pack292bytes, two-row1,341, three-row1,592, remainder302.

The helper returns scratch on both normal and exceptional paths. The normal
return precedes the odd-row remainder's read of original weights. The563byte
Tier1entry dispatcher retains its guards and two tail calls. Original smaller
matrix/shared methods and flags remain exact by the [compiled-scope review](build-20260923.md).
The code capture is independent of timing and cannot identify which versions
execute in a separately scored process.

All nine jobs and441resource observations pass; peak owned RSS448,860,160bytes.
Every worker/supervisor is terminal. Numerical closure:
`f5613c62f9bda671777ec0e0efca867a887400029339cc022cf7e8ef14ff0e7e`.
[Native review](codegen-review-20260923.json):
`8b769467aec1819cd4bbbf0fca60c3069c3d3e161e71ccb71429ee46292370a2`.
Raw evidence is under `artifacts/parakeet-wide-entry-first-use-numerics-amd-20260923`
and `artifacts/parakeet-wide-entry-first-use-codegen-20260923`.

This qualifies the candidate for the unchanged component comparison. It does
not establish application improvement or a new Microsoft ORT ratio.
