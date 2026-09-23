# Wide entry numerical and native qualification

All four numerical workers pass: selected/candidate with normal instructions
and AVX512 disabled. Each covers 68 groups and7,749,681 initial values, with complete
finite-bit/nonfinite checks and 1,560 independent coordinate checks. The original
66 groups remain unchanged. Two added Parallel(2) cases exercise the cloned
callback at 64 and 106 rows, both with 1,024 reduction/columns.

Both parallel cases preserve destination identity, poisoned repeated outputs,
retained allocating output, input arrays and surrounding guards. The 64-row case
splits 32/32 with zero scratch. The 106-row case splits 53/53: selected scratch is
zero and candidate scratch is 16,777,216 bytes across two destination calls,
matching four packs and returns. Every semantic observation agrees across
products and instruction widths.

The 51 complete emitted native bodies are retained and every branch label
resolves. The new entry is 5,625 bytes of first-use FullOpts; its packing helper
is 567 bytes of first-use FullOpts. All four arithmetic functions remain separate
calls and their normalized instruction streams exactly match the reviewed M52
copies: pack 292 bytes, two-row 1,341, three-row 1,592, remainder 302.

The helper returns scratch on both normal and exceptional paths. The normal
return precedes the odd-row remainder's read of original weights. The 563-byte
Tier1entry dispatcher retains its guards and two tail calls. Original smaller
matrix/shared methods and flags remain exact by the [compiled-scope review](build-20260923.md).
The code capture is independent of timing and cannot identify which versions
execute in a separately scored process.

All nine jobs and 441 resource observations pass; peak owned RSS 448,860,160 bytes.
Every worker/supervisor is terminal. Numerical closure:
`f5613c62f9bda671777ec0e0efca867a887400029339cc022cf7e8ef14ff0e7e`.
[Native review](codegen-review-20260923.json):
`8b769467aec1819cd4bbbf0fca60c3069c3d3e161e71ccb71429ee46292370a2`.
Raw evidence is under `artifacts/parakeet-wide-entry-first-use-numerics-amd-20260923`
and `artifacts/parakeet-wide-entry-first-use-codegen-20260923`.

This qualifies the candidate for the unchanged component comparison. It does
not establish application improvement or a new Microsoft ORT ratio.
