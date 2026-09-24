# Combined Parakeet improvements and release qualification

The [complete application comparison](application-20260924.md) admits the
recurrence and slice changes together: 9.477% lower latency, 1.700 times
Microsoft ORT latency. The [full graph comparison](graphs-20260924.md) fails
release admission: 30-token e5 is 6.769% slower and its candidate repeatability
control fails. Numerical checks and the other seven regression gates pass.
The [retained-clock review](e5-repeatability-20260924.md) localizes bursts but
does not establish their cause. The [runtime diagnostic](../../benchmarks/e5-runtime-diagnostic-results/report-20260924.md)
then confirms compilation continues into the measured window in both products.
The [focused e5 successor](e5-warmed-20260924.md) passes all unchanged gates.
The [qualified graph summary](qualified-graphs-20260924.md) binds that case
and the seven retained passing cases: all24controls and8regression gates pass.
Pyannote application/meetings, root integration and BENCHMARK.md remain pending.

After each relevant campaign closes, run these local reporting actions with
`C:/Python313/python.exe -X utf8 -B`:

    tests/parakeet/validated-composition-results/publish_release.py graphs
    tests/parakeet/validated-composition-results/publish_release.py pyannote
    tests/parakeet/validated-composition-results/publish_release.py root
    tests/parakeet/validated-composition-results/publish_release.py benchmark

The first three publish retained measurements and qualification evidence. They
verify closed files and refuse existing report outputs. The Pyannote timing
report is named `pyannote-application-20260924.md`; `pyannote-20260924.md` records
the separate model and public-result correctness checks.

Run `benchmark` only after the admitted source has passed normal root/package
qualification and has been committed locally. It checks matching product
identities, all admissions and every root source file, then updates the leading
full shortlist, evidence links and release description in `BENCHMARK.md`.
No publisher performs model inference or changes a performance verdict.
