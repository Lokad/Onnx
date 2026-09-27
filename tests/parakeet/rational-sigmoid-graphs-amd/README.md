# Matched graph regression for the fixed rational-sigmoid candidate

Compare current Core `8bb22038`, candidate `946ddfb6` and the same ORT 1.29.0
binary. The exact candidate has passed Parakeet correctness/application checks,
shared/e5 correctness and complete Pyannote correctness. Its isolated operator
screen remains rejected; double latency remains unresolved. No new implementation
variant, product build, consumer build or package/model download is introduced.

The three existing ReleaseBenchmark consumers are unchanged: `d827e3b9` for
ordinary cases, `0b228b2d` for 30-token e5, and `e437850d` for eight-token e5.
Retain their compiled instruction/flag proofs. The actual product compatibility
chain preserves the old public bindings and 3,979 method identities, with only
Sigmoid changed and one private helper added. All four exact graph exports contain
no Sigmoid nodes; retain the original extra candidate/current byte-equality check.
That static finding does not replace fresh numerical or performance qualification.

Run all 72 original workers for eight cases: five e5 inputs, DINOv3, ResNet50 and
GPT-2. Each case receives three verification workers and six timing processes in
current, candidate, ORT, ORT, candidate, current order. Preserve 6,000 warmups for
eight-token e5, 1,200 for 30-token e5, and 600 otherwise, then 180 measurements.
Keep all 73,512 calls, 8,640 measurements and 72 setup intervals.

Every native output must satisfy the unchanged scaled error bound 1e-4; current
and candidate managed arrays must match exactly. Ownership, finite values, shapes,
all <=5% regression gates and <=10% process-disagreement controls remain intact.
Do not discard clocks or relax a gate after execution. The original worker,
validators, native scripts, scorers and complete auditor are reused unchanged.

Stage from the latest qualified padding graph campaign, including its required
`source/global.json` working-directory input and all three runtimes. The old
missing-directory failure stays preserved. This is a fresh product comparison,
not a restart of that failure. Tools and inherited sources freeze at preparation.

Keep CPU 2 compute / CPU 0 monitoring, 11 GiB available RAM / 3 GiB tmpfs before
each job, 8 GiB owned RSS, 900 seconds/job, 1 GiB remaining RAM/tmpfs, 2 GiB campaign
files and four hours overall. Allow one VM workload at a time. Check memory margin
for the complete output set before staging; never lower a frozen resource limit.

From repository root, prefix commands with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/rational-sigmoid-graphs-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/rational-sigmoid-graphs-amd -p test_*.py
    tests/parakeet/rational-sigmoid-graphs-amd/run.py prepare
    tests/parakeet/rational-sigmoid-graphs-amd/run.py stage
    tests/parakeet/rational-sigmoid-graphs-amd/run.py launch
    tests/parakeet/rational-sigmoid-graphs-amd/run.py observe

Observe the same supervisor until terminal; collect and run audit.py once, with
audit stdout outside the artifact. Preserve every failed outcome. Local namespace:
`artifacts/parakeet-rational-sigmoid-graphs-amd-20260927`. Remote namespace:
`/dev/shm/lokad-parakeet-rational-sigmoid-graphs-20260927`.
Complete Pyannote application and actual-root/package qualification still follow
before source or BENCHMARK.md promotion.
