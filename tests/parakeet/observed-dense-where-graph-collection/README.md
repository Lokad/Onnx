# Complete the frozen graph collection

The running graph campaign uses a separate qualified runtime for 30-token e5.
The frozen collector enumerates `runtimes` but omits `runtimes-e5`, which the
unchanged auditor requires. It also omits the single staged `source/global.json`.
These are the only payload input directories missing from its collection list.

This adapter changes that one list in memory, adding the two directories.
The original transport, workers, consumers, inputs, timing and scoring remain
byte-for-byte unchanged. All original terminal-owner, payload, archive and
extracted-file checks remain. The correction has an exclusive review receipt,
an exact patch and an additional receipt inside the campaign for its final audit.
All eleven additional files must match the deployed payload after collection.

From the repository root, review once with:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/observed-dense-where-graph-collection/collect.py review

Only after the current supervisor and every owned worker are terminal, collect
once with:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/observed-dense-where-graph-collection/collect.py collect

Use this instead of the frozen `run.py collect`; do not run both. Then execute
the unchanged graph audit, keeping its stdout outside the campaign directory.
Do not edit prepared tools, repeat workers, score partial output or reuse a
partial collection. The adapter provides no inference or performance result.
