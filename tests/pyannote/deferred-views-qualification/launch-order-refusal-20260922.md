# Parakeet launch ordering correction

The first `parakeet.py run` invocation exited 1 before starting inference:
`FileNotFoundError` for
`artifacts/pyannote-deferred-views-parakeet-20260922/prepared.json`.
Its preparation invocation had yielded live session13477, rather than exited.
The parent incorrectly continued to the dependent command without reaping it.

The refusal occurred on the first read in `run()`, before any process-state
file, worker launch, model load or numerical check. Preserve the preparation
and wait for its actual successful exit. Only then execute `run()` against the
newly completed preparation. This changes prerequisite state; it does not retry
a failed numerical or timing trial. The original source and gates stay fixed.

The correction is to inspect each tool result and wait on a returned session
before starting dependent work, including when calls share one orchestration
cell. Successful independent audits are not repeated.
