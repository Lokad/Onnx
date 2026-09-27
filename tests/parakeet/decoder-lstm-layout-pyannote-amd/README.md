# Pyannote correctness for the fixed LSTM layout candidate

Qualify candidate Core `ad97b4ad` / Data `29b3f633` against qualified current
Core `0d224bcf` / Data `a3745392`. This unchanged candidate passes complete
Parakeet correctness, its independent application comparison (2.444120648%
gain, candidate/ORT 1.187748548) and shared/e5 correctness. This lane introduces
no implementation variant and measures no performance. Preserve the rejected
component screen and the runtime diagnosis that followed it.

Reuse GraphQualification `d78c45b9` from the latest qualified Pyannote campaign.
Its product-identity arguments and guards remain unchanged. The completed model
compatibility review reconciles 3,981 original methods through the actual root:
two Core methods change (Lstm and GraphLstmPacking.Prepare), two internal reader/
getter methods are added, and all 697 Data methods remain identical. Data file
hashes differ between these builds; use their exact identities. Original method
flags, public bindings and assembly metadata remain unchanged. The consumer is
reused byte for byte. Its original inspector proved 95/96 methods unchanged with
only Main identity-argument edits, but recorded no method flags; no retrospective
consumer-flags claim is made.

Run the original three serial jobs: identity probes, selected correctness and
candidate correctness. Four wrong-Core/wrong-Data probes must reject before
creating outputs and their PID/birth identities must be terminal. Each product
then checks 18 arrays / 2,917,107 values and sixteen complete public requests.
Preserve original finite-output, shape, timeline, speaker, centroid, ownership
and native 1e-4 gates. Require byte-identical current/candidate arrays and exact
complete public results, including centroids.

The worker, protocol, native/public validators and identity probes are unchanged.
Audit changes only its candidate label. Remote preparation adapts namespace and
label bindings, and checks needed retained files in older namespaces whose
obsolete source/output copies were retired. It still verifies original collection
and payload metadata, terminal owners, every external input and every needed
assets/graph-reference/runtime/built link. Fresh model/app/shared initial payloads
remain fully verified. No build, restore, model download or source promotion occurs.

Keep CPU2 compute / CPU0 monitoring, one VM workload, 11 GiB RAM / 3 GiB tmpfs
preflight, 8 GiB RSS, >=1 GiB RAM/tmpfs remaining, 900 seconds/job, 1 GiB/job output
and 2 GiB campaign files. Verify real headroom before staging. Reuse retained
assets, consumers and products through exact hardlinks.

From repository root, prefix with C:/Python313/python.exe -X utf8 -B:

    tests/parakeet/decoder-lstm-layout-pyannote-amd/consumer_scope.py
    tests/parakeet/decoder-lstm-layout-pyannote-amd/consumer_reuse.py
    -m unittest discover -s tests/parakeet/decoder-lstm-layout-pyannote-amd -p test_*.py
    tests/parakeet/decoder-lstm-layout-pyannote-amd/run.py prepare
    tests/parakeet/decoder-lstm-layout-pyannote-amd/run.py stage
    tests/parakeet/decoder-lstm-layout-pyannote-amd/run.py launch
    tests/parakeet/decoder-lstm-layout-pyannote-amd/run.py observe
    tests/parakeet/decoder-lstm-layout-pyannote-amd/run.py collect
    tests/parakeet/decoder-lstm-layout-pyannote-amd/audit.py

Freeze before preparation; execute mutations once and follow the original owner
until terminal. Collect/audit once and keep audit output outside the artifact.
Never replay a completed campaign. Local namespace:
artifacts/parakeet-decoder-lstm-layout-pyannote-amd-20260927;
VM /dev/shm/lokad-lstmlayout-pyannote-20260927.

Graph and complete Pyannote application comparisons, then actual-root/package
qualification, remain necessary before source or BENCHMARK.md promotion.
