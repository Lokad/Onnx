# Parakeet admitted-source composition: build and tensor qualification

The combined candidate passes compiled implementation review and **369/369 tensor
cases in each instruction mode**, without skips. This composes the independently
admitted recurrent preparation and slice materialization changes; it adds no
new optimization. Full Parakeet, application and release qualification remain.

The compiled review preserves all 3,249 unaffected recurrence methods and requires
both changed/new slice-copy bodies to match the measured slice candidate exactly.
Public surface, implementation flags and original fallback remain unchanged.
The exact measured recurrence Data assembly is reused.

The first combined build passed that review but failed one of 369 normal tensor
tests: the existing source rule prohibits optional parameters, and two new
recurrence test helpers had defaults. V2 removes those defaults and supplies
the same values at all 13 call sites. Every product source byte and all 3,251
compiled Core method bodies/flags match V1. The original failure is retained;
no timing run was repeated or scoring rule changed.

Each corrected suite covers 343 existing cases, 25 slice-copy contracts and one
consumed-Core/actual-instruction-mode check. Ordinary mode and
`DOTNET_EnableAVX512=0` both pass. All build/test owners are terminal; all 40 test
resource samples pass, with peak owned RSS 375,062,528 bytes.

Candidate Core: `37c243756bbe5e5d563e79627a0a4d20ff027598801849da9606549c7ac60286`.
Data: `cc37b19eb41cf728061c35bcdb7e06a6ab4d370bfd4c47555eec86c86b2ef6d6`.
Source: `af68e6c2c28f5794f3eece39bd9e78fc6809e2ca8925a5616ff2956c405061c4`.
Closure: `bb578d7ad6dfeaf989726476d6d7864438e602a41c7727a99ee05cb8c9eb71da`.
Original failure: `a51a8a8dc4d7cac32b16c61abebfc17b09bc954544d4443821aabee66a4236d6`.
Raw evidence: `artifacts/parakeet-validated-composition-build-amd-v2-20260924`.

[Full identities, reviews and failure record](qualification-20260924.json).
No combined speedup or release readiness is claimed here.
