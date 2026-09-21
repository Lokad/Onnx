# Whisper public contracts on AMD

This finite replay uses exactly the portable consumer and private product DLLs
already qualified by the [Windows replay](../memory-contracts-v2/local-results-20260921.md).
It checks thirteen completed requests, sixteen refusals or cancellations, actual
concurrent speech, subsequent recovery and both complete decoder weight snapshots.
The [original sharing failure](../weight-sharing/failure-20260920.md) remains intact.

From the repository root, use `C:/Python313/python.exe -X utf8 -B`:

1. `tests/whisper/memory-contracts-amd/prepare.py` creates a local payload without
   running a model or changing the VM.
2. After the separate sharing endurance campaign is closed and independently
   verified, `deploy.py` verifies existing data and launches one new finite worker.
3. `observe.py` reports the existing process identities; it never starts work.
4. Once every recorded process is terminal, `collect.py` retrieves all raw records.
5. `audit.py` independently checks application decisions, recording timelines,
   original initializer hashes, resource and thread-affinity samples.
6. On success, `close.py` writes the complete report and seals all evidence;
   `verify.py` independently recomputes displayed results, overlap, initializer
   checks and resource totals. A failed run must retain its failure and must not
   use these success-only report tools.

The new artifact is `artifacts/whisper-memory-contracts-amd-20260921`. Every writer
creates new evidence and refuses an existing destination. Reuse verified hardlinks
only for unchanged dependencies and PCM. Model files remain at their existing
canonical paths. There is no build, native-reference inference or model download.

Normal .NET 10.0.8 and CPU 2 before CLR startup are required. The supervisor samples
process and thread affinity, RSS, available memory and disk every 0.5 seconds. Limits
are 1,800 seconds, 14 GiB RSS, 1 GiB available memory and 32 MiB disk, with 13 GiB
available memory and 64 MiB disk required before launch. No forced collection or
LOKAD/DOTNET/COMPlus override is used. This qualifies a private memory candidate;
it is not a matched ORT latency measurement or production integration.
