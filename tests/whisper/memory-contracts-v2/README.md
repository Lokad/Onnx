# Whisper recording, concurrent speech and recovery contracts

This corrected private consumer preserves all thirteen completed requests and
sixteen refusal checks from the prepared public contract replay. It adds no
product behavior: the change replaces a faulty all-metadata equality assumption
with the independently tested two-name transition policy and saves both complete
weight snapshots before validation.

`prepare.py` builds against the exact backend-tested Core/Data binaries.
`run_local.py` freezes and executes the finite Windows replay; `observe.py` checks
its existing process identities without restarting it. `audit.py` verifies all
public/native decisions, reconstructed recording timelines, overlapping real
speech requests, exact recovery, held outputs, weight bytes and resource samples.
Run the auditor only after the worker and supervisor are terminal.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root. The artifact is
`artifacts/whisper-memory-contracts-v2-20260921`. The first unused preflight refusal
remains in the separate original artifact. These results do not constitute a
matched ORT latency measurement or production integration.
