# Shared-model and e5 qualification for owned attention weights

Application closure 600e9e67 admits the unchanged candidate: all 63 controls and
21 gates pass, with 1.128490% matched gain and candidate/ORT 1.147329. APP_DIGEST
binds that actual complete result; parity remains open.
The completed Parakeet model closure is 2d64fad9, focused contracts a293403a and
exact-model census ef4e41ec. Current Core a6f7d9f9 and candidate ee5218db differ
only in PrepareOwnedMatMulWeights; 3,287 other Core methods and 697 Data methods,
flags and public bindings are unchanged. No consumer or product is rebuilt.

Reuse original Replay a50d3e96 and native fixtures. Four original serial processes
run selected-shared, selected-e5, candidate-shared and candidate-e5. Each product
checks 166 arrays / 5,000,814 values, all five e5 shapes, repeated context reuse,
memory policies, immutable inputs and independently held outputs. Require scaled
native error <=1e-4, finite values, exact shapes and bit-exact current/candidate
arrays. No Data assembly is loaded and no performance is scored.

The worker, protocol, checks and auditor match decoder-lstm-layout-shared-amd
byte for byte. The preparation body retains every fixture; remote preparation
changes only model/application namespaces. The local eligibility adapter binds
the actual single-method scope and must reject failed application controls.

Keep CPU2 compute / CPU0 monitoring, 11GiB available RAM / 3GiB tmpfs preflight,
8GiB owned RSS, 1GiB free RAM/tmpfs and 900 seconds/job. Reuse existing assets.
After application admission, bind its real closure digest, run the seven local
prerequisite tests, freeze tools and use C:/Python313/python.exe -X utf8 -B:
tests/parakeet/attention-owned-shared-amd/run.py prepare, stage, launch, observe.
Follow the same owner to terminal, collect once and audit.py once. Never replay.

Artifacts: artifacts/parakeet-attention-owned-shared-amd-20260928.
VM: /dev/shm/lokad-attention-owned-shared-20260928. Pyannote, graph and actual-root
qualification remain required afterward. Root product and BENCHMARK.md remain
unchanged until release qualification completes.
