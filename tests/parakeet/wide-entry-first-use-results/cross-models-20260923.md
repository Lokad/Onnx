# Wide-entry candidate: complete shared/e5 and Pyannote correctness

Both products pass all complete numerical and public checks. Every candidate
tensor and public result matches the fresh selected product exactly.
These correctness results do not establish graph or Pyannote application performance.

| Scope per product | Arrays | Values | Public calls | Maximum native scaled error |
| --- | ---: | ---: | ---: | ---: |
| Shared graphs and five e5 inputs | 166 | 5,000,814 | — | 1.9803643226623535e-05 |
| Complete Pyannote graphs and public diarization | 18 | 2,917,107 | 16 | 3.7282705307006836e-05 |

Both roles independently retain the original native error bound of0.0001.
Shared/e5 includes repeated contexts, facade use, memory policy, unchanged
inputs and held outputs. Pyannote includes complete segmentation and embedding
arrays, all public speaker assignments, timelines, statuses and centroid values.

The shared Replay consumer is reused unchanged. The rebuilt Pyannote consumer
changes only its expected Data-assembly hash:95 methods remain exact and one
method differs by that literal only; the public surface is unchanged.

Shared/e5: 4 completed jobs, 160 resource samples; peak RSS 2,659,418,112bytes. All monitoring gaps are below10seconds.
Pyannote: 5 completed jobs, 486 resource samples; peak RSS 1,386,696,704bytes. All monitoring gaps are below10seconds.

All recorded owners are terminal. Pyannote waited at its unchanged12GiB
preflight between completed workers. With no inference child, the exact
supervisor was suspended at that boundary;1,822 duplicate files in closed
campaigns were linked after verifying14,298 hashes, freeing219,025,408bytes
of tmpfs. The active payload remained exact and the same supervisor resumed.
No inference was stopped or repeated, and no resource bound changed.

Actual products: selected Core521bae17/Dataf3b9aa81, candidate Core672e5f30/Data065b7a7f.
AMD CPU2 computes and CPU0 monitors, .NET10.0.8, ordinary runtime flags.
Root source and BENCHMARK.md remain unchanged pending performance and release checks.

[Complete results, identities, resources and monitoring gaps](cross-models-observations-20260923.json).

Shared/e5 closure: `689c0c0966a4d1ba397210ecdcb037b13df653b01560f9cd704273ee92f0a69a`.
Pyannote closure: `8133a7ffe446e7e3f494e5db041c12f6e629e017026f4630989befa1308e8a80`.
