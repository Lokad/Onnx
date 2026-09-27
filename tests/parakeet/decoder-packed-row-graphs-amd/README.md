# Qualify the fixed prepared-row change on all eight graph cases

Compare current Core `65f15a41`, candidate `af19b3b4` and the same ORT 1.29.0
binary. Require completed Parakeet model/application, shared/e5 and Pyannote
correctness before preparation. Preserve the rejected mixed-layout operator
screen and failed first-call-only explanation. This is one unchanged candidate;
no product build, consumer build, variant selection or model download occurs.

Reuse the three exact ReleaseBenchmark consumers: `d827e3b9` for ordinary cases,
`0b228b2d` for 30-token e5 and `e437850d` for eight-token e5. Preserve their closed
compiled instruction/flag proofs. The 3,980-method product compatibility chain
connects the previously measured graph product to the current root and candidate.
Only two prepared-MatMul dispatch methods change and one internal row kernel is
added; all original public bindings and method flags remain identical.

Reuse the existing four-export/eight-case census solely for model identity and
case coverage. Its old finding about Sigmoid absence is irrelevant to this change;
it supplies no new runtime-dispatch claim. Fresh numerical and performance checks
are required regardless of whether a particular graph reaches the changed path.

Run all 72 original workers: five e5 inputs, DINOv3, ResNet50 and GPT-2. Each case
receives three verification workers and six timing processes in current,
candidate, ORT, ORT, candidate, current order. Keep 6,000 warmups for eight-token
e5, 1,200 for 30-token e5, and 600 otherwise, then 180 measurements. Retain all
73,512 calls, 8,640 measured clocks and 72 setup intervals.

Native scaled error remains <=1e-4, and every managed candidate array must match
current bytes exactly. Preserve finite values, shapes, ownership, <=5% regression
gates and <=10% process-disagreement controls. Reuse the original worker, native
scripts, validators, scorers and complete auditor unchanged. Never discard clocks
or relax a gate after execution.

Stage from the closed rational-sigmoid graph campaign, including its required
`source/global.json` working-directory file and all three runtimes. The original
missing-directory failure remains preserved. Freeze tools and inherited sources
at preparation. Keep CPU 2 compute / CPU 0 monitoring, 11 GiB available RAM /
3 GiB tmpfs before each job, 8 GiB owned RSS, 900 seconds/job, 1 GiB remaining
RAM/tmpfs, 2 GiB campaign files and four hours overall. Confirm headroom for the
complete output set before staging; only one VM workload runs at a time.

From the repository root, prefix commands with `C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/decoder-packed-row-graphs-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/decoder-packed-row-graphs-amd -p test_*.py
    tests/parakeet/decoder-packed-row-graphs-amd/run.py prepare
    tests/parakeet/decoder-packed-row-graphs-amd/run.py stage
    tests/parakeet/decoder-packed-row-graphs-amd/run.py launch
    tests/parakeet/decoder-packed-row-graphs-amd/run.py observe

Observe the same owner until terminal, then collect and audit once, placing audit
stdout outside the artifact. Preserve every failed result. Local namespace:
`artifacts/parakeet-decoder-packed-row-graphs-amd-20260927`; VM namespace:
`/dev/shm/lokad-parakeet-decoder-packed-row-graphs-20260927`.
Complete Pyannote application and actual-root/package qualification still follow
before source integration or BENCHMARK promotion.
