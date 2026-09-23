# M43 shared-model, e5 and Pyannote correctness

The candidate passes fresh complete correctness checks for the shared graph
shortlist, all five e5 inputs and Pyannote. All candidate tensors and public
results match the selected product exactly. These are correctness results;
application and graph performance regressions are separate gates.

| Scope per product | Arrays | Values | Public calls | Maximum native scaled error |
| --- | ---: | ---: | ---: | ---: |
| Shared graphs and five e5 inputs | 166 | 5,000,814 | — | 0.000019803643226623535 |
| Complete Pyannote graphs and public diarization | 18 | 2,917,107 | 16 | 0.000037282705307006836 |

Both products stay below the original 0.0001 native error bound. Shared/e5
checks cover repeated execution, memory policy, facade/context use, unchanged
inputs and retained outputs. Pyannote checks include every segmentation and
embedding output, unchanged inputs, held outputs, exact speaker decisions,
timelines, statuses and centroid coordinates.

Shared/e5 uses the unchanged consumer in four fresh processes. All 157 resource
observations pass; peak owned RSS is 2,520,670,208 bytes. Pyannote's consumer
rebuild changes only its expected Data-assembly hash: 95 methods are unchanged
and the remaining method differs by that literal alone. All five jobs and 473
resource observations pass; peak owned RSS is 1,186,140,160 bytes. Every recorded
owner is terminal.

The shared staging controller exited before saving its local receipt, after
the VM had completed staging. A separate recovery verified the same payload
and absence of workers, then retrieved only that receipt. No input mutation,
inference or timing was repeated.

Shared/e5 closure: `5b3014960c84312903b62739ecc51922f5e45961b0a1e022c70b264f5771be6c`.
Pyannote closure: `93354608bf15c3d66e9f5c5730afb4ec48d12f972cdf34ae44f14f6d98b942ab`.
The actual products remain Core `521bae17` / Data `f3b9aa81` for selected and
Core `c00a25b4` / Data `9b623be9` for candidate. Raw artifacts are retained under
`artifacts/parakeet-first-use-kernels-shared-amd-20260923` and
`artifacts/parakeet-first-use-kernels-pyannote-amd-20260923`.
