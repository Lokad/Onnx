# Check the fixed LSTM layout on the original eight graph cases

Compare current Core `0d224bcf`, candidate `ad97b4ad` and the same ORT 1.29.0
binary. Require completed Parakeet model/application, shared/e5 and Pyannote
correctness before preparation. The independent Parakeet application comparison
passes, saving 2.444120648%, with candidate/ORT 1.187748548. Preserve the failed
component repeatability screen and its subsequent runtime diagnosis. This is
release qualification of one fixed candidate, with no new implementation variant.

Reuse exact ReleaseBenchmark consumers `d827e3b9` for ordinary cases, `0b228b2d`
for 30-token e5 and `e437850d` for eight-token e5. Preserve their compiled method
and flag proofs. The compatibility chain reconciles 3,981 original product
methods through the actual root: Lstm and GraphLstmPacking.Prepare change, two
internal projection-reader/getter methods are added, all 697 Data methods stay
exact, and original public bindings, method flags and metadata are unchanged.
These graph consumers load Core only. No product or consumer build occurs.

The closed four-export/eight-case census establishes model identity and case
coverage. It supplies no new runtime-dispatch claim. Keep the original 72 jobs:
five e5 inputs, DINOv3, ResNet50 and GPT-2, with three verification workers and
six timing processes per case in current, candidate, ORT, ORT, candidate, current
order. Keep 6,000 warmups for eight-token e5, 1,200 for 30-token e5, and 600
otherwise, then 180 measurements. Retain all 73,512 calls, 8,640 measured clocks
and 72 setup intervals.

Reuse the original workers, numerical validators, scorers and complete auditor
unchanged. Require finite outputs, native scaled error <=1e-4, exact candidate/
current arrays, ownership, <=5% regression and <=10% process disagreement.
No clock filtering, scoring changes or model downloads are permitted.

Reuse retained references, three runtimes and source/global.json from the latest
qualified prepared-row graph campaign. Verify exact collection/payload metadata,
terminal owners, every external input and every linked file; older obsolete
source/output copies were retired. All new prerequisite payloads remain intact.
Freeze the adapter before preparation. Keep CPU2 compute / CPU0 monitoring,
11 GiB available RAM / 3 GiB tmpfs preflight, 8 GiB owned RSS, 900 seconds/job,
1 GiB RAM/tmpfs remaining, 2 GiB campaign files and four hours overall. Only one
VM workload may run; check headroom for the complete output set before staging.

From repository root, prefix commands with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/decoder-lstm-layout-graphs-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/decoder-lstm-layout-graphs-amd -p test_*.py
    tests/parakeet/decoder-lstm-layout-graphs-amd/run.py prepare
    tests/parakeet/decoder-lstm-layout-graphs-amd/run.py stage
    tests/parakeet/decoder-lstm-layout-graphs-amd/run.py launch
    tests/parakeet/decoder-lstm-layout-graphs-amd/run.py observe

Observe the same owner to terminal, then collect and audit once with audit output
outside the artifact. Never replay a completed campaign. Local namespace:
artifacts/parakeet-decoder-lstm-layout-graphs-amd-20260927; VM namespace:
/dev/shm/lokad-lstmlayout-graphs-20260927.

Complete Pyannote application and actual-root/package qualification remain
necessary before source or BENCHMARK.md promotion.
