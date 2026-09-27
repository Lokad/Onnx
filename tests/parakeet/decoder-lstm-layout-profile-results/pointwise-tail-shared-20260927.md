# Pointwise candidate preserves shared-model and e5 outputs

All four correctness jobs pass: **332 arrays / 10,001,628 values** across current
Core `47984318` and the unchanged candidate `7cac6788`. Every candidate output is
byte-identical to the corresponding current output. Input immutability, retained
output ownership, memory policies and execution-context reuse pass.

Each product checks 106 shared-model arrays and 60 e5 arrays, covering 8 tokens,
30 tokens, 30 tokens padded to 128, 128 tokens and 512 tokens. Maximum scaled
error against the original Microsoft ORT arrays is 0.00001980364323 for shared
models and 0.000001639127731 for e5, below the unchanged 0.0001 bound. All finite
values and shapes are checked. Replay `a50d3e96`, models and native fixtures were
reused without builds or downloads.

The complete original worker, protocol, validator and auditor are unchanged.
All 151 resource samples pass, with peak owned RSS 2,810,929,152 bytes.
Supervisor 1228226 / birth1790535565.96 and all children are terminal; 565 files
were collected once and independently audited. Execution durations are
correctness costs, not performance scores.

The [admitted Parakeet application gain](pointwise-tail-app-20260927.md) remains
2.132562%, with a 1.155498 candidate/ORT ratio. Pyannote correctness, graph and
Pyannote application regression checks, portable boundary coverage and actual-root
tests/package qualification remain required before source or release-table
promotion. The two drafted portable facts cover 111 raw geometries, including
widths 0–31; they have not yet been compiled or executed.

[Identities, numerical maxima and resources](pointwise-tail-shared-observations-20260927.json),
[protocol](../pointwise-tail-shared-amd/README.md).

Closure: `fe35674de649414c520cc73812d485c93cde6aadbc05db0c17ab94ca883e8c1b`.
Raw evidence: `artifacts/parakeet-pointwise-tail-shared-amd-20260927`.
