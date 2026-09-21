# Prospective request-context comparison

Run predecessor/candidate/ORT/ORT/candidate/predecessor, sequential fresh
processes on Windows i7-14700KF CPU 2. Each runs the original four-fixture
pyannote manifest: one warmup and three measured passes, 96 total requests.
Use the unchanged complete public application consumer and native adapter.

Both managed roles use Core `469cb2d6`; predecessor Data is `e7fe1668`, candidate
Data is `1d346664`. The Data candidate must first pass its full suites, all
dialogue calls, both ten-minute meetings and recovery. No build is timed.

Every role must have a ratio of larger to smaller process mean <=1.10 on the
full 30-second request and <=1.20 on each fixture. All observations remain in
the report even if controls fail. Performance admission requires all controls,
candidate/predecessor full-request ratio <=0.97, and no fixture ratio >1.05.
Reduced allocation by itself is insufficient for performance admission.

Use normal .NET 10.0.12 and ORT 1.29.0 with sequential CPU execution, one thread
and spinning disabled. Each worker has a 10 GiB available-memory preflight,
8 GiB RSS ceiling, 1 GiB minimum available RAM, 20 GiB free disk minimum,
1 GiB output ceiling and 1,800-second runtime ceiling. Preflight waits at most
900 seconds. Preserve every resource sample, input/output check and actual
PID/creation-time identity. Never overlap local inference workers.

This is a descriptive local comparison, with no calibrated confidence interval,
AMD extrapolation or automatic production promotion. Keep the frozen e5 and
queued primary AMD pyannote jobs unchanged. No unchanged retry after failure.
