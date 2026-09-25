# First full graph comparison for M78

Compare qualified release Core `f95a13c5` with admitted isolated M78 `49901366`,
plus fresh Microsoft ORT processes. M78's Parakeet application baseline was M73;
this graph comparison explicitly uses the qualified release. Both products have
passed fresh shared/e5 and Pyannote correctness. Reuse their exact binaries from
the closed shared-model stage; build nothing.

All eight cases remain: e5 at 8, 30, 30 padded to 128, 128 and 512 tokens,
DINOv3, ResNet50 and GPT-2. The qualified consumer `0b228b2d` retains 1,200
warmups for 30-token e5; `d827e3b9` retains 600 for the other seven cases.
Each retains 180 measured calls. Native scripts and runtime flags are unchanged.

Run all 72 processes: three numerical processes per case, then six timing
processes in release, candidate, ORT, ORT, candidate, release order. Preserve
41,112 calls, 8,640 measurements, every setup interval, exact candidate/release
outputs, native scaled error at most 1e-4, finite values, input immutability and
held-output ownership. Every engine's repeatability ratio must be at most 1.10;
every candidate case must stay within 5% of the qualified release.

The earlier M73 run failed two repeatability controls. Its failed closure and
the two closed diagnostic observations remain prerequisites/evidence; they are
not admitted by this new comparison. The bounded frequency observation did not
reproduce the timing spread. No warmup, score or admission threshold changes.
If M78 fails its first comparison, retain the verdict; do not rerun unchanged.

The worker, numerical checks, full compiled-consumer checks, scorers, tests and
independent auditor are byte-identical to the qualified graph lane. Only product
provenance and transport namespaces change. `consumer_scope.py` checks all 14
unchanged files. The original 11 GiB RAM / 3 GiB tmpfs preflight, 8 GiB RSS,
1 GiB remaining RAM/tmpfs, 1 GiB output/job, 2 GiB stage, 900 seconds/job and
four-hour total limit remain. CPU 2 computes and CPU 0 monitors; no other VM work.

From the repository root, prefix each command with
`C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/packed-final-row-graphs-amd/consumer_scope.py
    -m unittest discover -s tests/parakeet/packed-final-row-graphs-amd -p "test_*.py" -v
    tests/parakeet/packed-final-row-graphs-amd/run.py prepare
    tests/parakeet/packed-final-row-graphs-amd/run.py stage
    tests/parakeet/packed-final-row-graphs-amd/run.py launch
    tests/parakeet/packed-final-row-graphs-amd/run.py observe
    tests/parakeet/packed-final-row-graphs-amd/run.py collect
    tests/parakeet/packed-final-row-graphs-amd/audit.py

Prepare only with all exact prerequisites verified. Observe the same owner until
terminal. Collect/audit once; keep audit stdout outside the artifact directory.
Local namespace: `artifacts/parakeet-packed-final-row-graphs-amd-20260925`.
VM namespace: `/dev/shm/lokad-parakeet-packed-final-row-graphs-20260925`.
