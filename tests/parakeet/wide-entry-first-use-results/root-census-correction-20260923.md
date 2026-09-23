# Root qualification: omitted AVX512 test census

The first normal-root run is **incomplete and not admitted for release**.
Its supervisor stopped in the disabled-instruction test checker after eleven
successful processes. Package creation and the independent consumer did not run.

The checker incorrectly expected ten existing `Exp512Tests` cases to pass with
AVX512 disabled. All five test methods explicitly skip unless
`Vector512.IsHardwareAccelerated` and `Fma.IsSupported`: four facts plus six
tail lengths produce ten cases. Their source is byte-identical to the selected
release (`1cb029e3`, 4,644 bytes). All ten pass in ordinary mode and skip in the
disabled mode. This is a checker omission; no product test reported a failure.

| Complete suite | Pass | Skip | Fail |
| --- | ---: | ---: | ---: |
| Ordinary backend | 3,449 | 41 | 0 |
| Ordinary tensor | 343 | 0 | 0 |
| AVX512-disabled backend | 3,359 | 131 | 0 |

The exact disabled census comprises 93 hardware-dependent pass-to-skip changes
and three unsupported-hardware skip-to-pass changes. Every other outcome must
remain identical to selected. The corrected checker names the five omitted
methods explicitly and retains complete per-test comparison; it cannot admit
an arbitrary additional skip or outcome change.

All collected file hashes, process identities, monitoring gaps and resource
limits verify. The normal build preserves all 3,189 Core and 697 Data methods,
implementation flags and public interfaces against the measured candidate.
The V1 tools, failed verdict and all observations are preserved unchanged.

A fresh V2 root qualification repeats all sixteen normal build/test/package
jobs with only the declared checker correction and namespace/dependency changes.
No scored performance run is repeated. Its independent closure is required
before publishing the candidate as the release.

[Complete incident evidence](root-census-correction-20260923.json).
V1 closure: `14f5373cead061c8cc5de1847d1cd0fe0b76db6b0dca73137f83b3ca41b744cf`.
