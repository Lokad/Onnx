# Proof metadata clarification

The frozen consumer labels every one-row case `nonzero_destination: true`.
That descriptive field is incorrect for `row-1-1-special` in both eligible
instruction modes: its deterministic destination is positive zero. The
corresponding finite case starts at -973/997 and covers nonzero accumulation.
Across the 48 one-row cases, 46 destinations contain nonzero values and two
are zero. Complete-route cases intentionally start with zero destinations.

This does not change the numerical comparisons, guard checks, immutable-input
checks or measured zero-allocation assertions: those execute against the actual
arrays and all pass. The raw records and original closure remain preserved.
Treat the field as incorrect metadata rather than evidence of a particular
destination's contents. Any future consumer must derive it from the actual
initial destination; do not modify or rerun the completed frozen proof.

[Proof and coverage](proof-20260925.md). Source: the frozen consumer's
`Values` and `Row` methods in
`artifacts/parakeet-packed-final-row-proof-amd-20260925/capture-collected/source/Program.cs`.
