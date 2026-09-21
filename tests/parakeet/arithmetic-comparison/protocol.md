# Prospective complete Parakeet arithmetic comparison

Compare production Core `d1f86a73`, qualified partial-sum Core `f2c292cb` and
Microsoft ORT 1.29.0 on Windows i7-14700KF, CPU2, normal .NET 10.0.12.
Both managed roles retain identical Data, consumers and the original 256 MiB
encoder packing budget; only Core differs. This candidate already passes
complete native Parakeet, shared-model, affected pyannote and full-suite checks.

Use the original `AudioBenchmark.dll` and `native.py` / `native_adapters.py`,
with the unchanged twenty-clip corpus totaling 213.265 seconds. Run six fresh
processes, sequentially, in this order:

    production, candidate, ORT, ORT, candidate, production

Each performs one complete warmup pass and three measured passes. Retain all
480 requests: 120 warmup and 360 measured. Within each process, average the
three measured times for each clip and sum the twenty means for corpus latency.
Average the two process means equally for each role. Publish per-clip and
per-process observations as well as aggregate ratios; remove no outliers.

For every role, require corpus process-mean max/min at most 1.10 and every
clip's process-mean max/min at most 1.20. All controls must pass before
attributing a performance difference. The candidate's fixed performance screen
requires aggregate corpus latency at most 0.95 of production and every aggregate
clip latency at most 1.05 of production. Failure is retained without changing
these limits or rerunning an unchanged campaign. These are descriptive screens,
not confidence intervals or proof of parity.

Both engines time feature extraction, graphs, greedy decoding and owned output
construction. Model loading, input-file access and external validation remain
outside both timers. Preserve every original public decision, repeat, input and
held-output check. Production's three known native tensor failures remain
separate from timing; the candidate passes those original numerical fixtures.

Each worker inherits CPU2 before startup. Require 14 GiB available RAM before
launch, at most 12 GiB process RSS, 1 GiB available RAM during execution, 20 GiB
free disk, at most 1 GiB output and 1,800 seconds. Preflight waits at most
900 seconds. The existing monitor records samples before testing limits and
stops only its owned PID/birth identities. No unrelated process is modified.

The earlier 512 MiB budget trial failed its available-memory guard and remains
closed. This experiment tests different arithmetic at the smaller original
budget; its qualified candidate public RSS peak is about 8.50 GiB. All guards
remain active. Any resource failure invalidates a complete timing comparison.

Run once from the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/arithmetic-comparison/prepare.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/arithmetic-comparison/run.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/arithmetic-comparison/audit.py

The audit begins only after the run supervisor and all workers are actually
terminal. No finisher writes concurrently with final closure. Preserve failed
attempts; do not run the success audit on incomplete evidence.

Pyannote retains priority on the AMD VM. Neither this comparison nor a passing
local screen changes the frozen e5/pyannote campaigns, production defaults or
the requirement for AMD qualification.
