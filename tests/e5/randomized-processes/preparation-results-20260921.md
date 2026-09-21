# e5 full campaign preparation — 2026-09-21

The full payload is prepared locally. **No e5 VM inference has started.** The
ongoing Whisper comparison remains the sole VM inference workload.

Tools were committed in `135ff2a`, followed by the NumPy binding correction
`c3f8a35`. Preparation copies the previously qualified producer unchanged; it
does not rebuild the core or repeat its inference qualification. Both assignment
schedules remain the original one-shot draws.

The first preparation stopped before creating a payload: NumPy was in the
interpreter's user site, outside the three additional audio package paths.
The corrected preparation binds those already recorded NumPy files too. Both
logs remain under `artifacts/e5-randomized-processes-20260921`.

| Prepared item | Bytes / count | SHA256 |
|---|---:|---|
| `prepared.json` | 598 B | `bd44ec5c81054c8af6810693d1282c381e562453ac3eb0f2e5983d2b0154b528` |
| Local template | 1,119,986 B | `e2e54890c8e286305e0957bbaf2dc873ef8884297dc81db0c18399a194f0ef0e` |
| Payload archive | 2,396,483 B | `508a1893f30df3acc26875fe8103ee091b97cd0beea16d6528799253b1346543` |
| Qualified worker | 48,640 B | `09872f68453bda9c679f8a31d6a3801f78d69f0ce56940cd6b400b4b310248e7` |
| Qualified core | 727,040 B | `d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4` |

The archive has 30 files totaling 4,920,117 uncompressed bytes. The template
binds 192 runtime files and 914 Python files, including 877 NumPy files. Actual
predecessor process identities and the resolved Linux Python executable are
added only after Whisper closes. Staging fetches the exact Linux manifest
bytes, preserving its digest across Windows newline conventions.

Eleven mathematical, contract and reporting tests pass. The reporting fixtures
exercise both complete 1,800-worker/334,080-call shapes without model inference;
80-digit Decimal recomputation checks the Fieller intervals independently.
Malformed summaries, duplicate cases and wrong table rows are rejected. After
tightening exact contrast and process-mean checks, both reporting tests pass
again. Three controller tests also pass: failed A/A reports without comparison,
passing A/A launches comparison once, and predecessor/collection failure blocks
comparison. No test contacts the VM.

Before deployment, the maximum single-cohort observed variance share is fixed
at 20%. This is an additional refusal condition; passing does not establish the
regularity of unobserved potential outcomes. Counts, assignments, numerical
gates and performance thresholds are unchanged. Each phase still needs at least
15 hours of conditioning across 1,800 fresh processes.

The [run instructions](README.md) describe guarded deployment, streamed
collection, full raw-output checks, independent report verification and updates
to `BENCHMARK.md`. The controller waits for Whisper's completed verification,
runs A/A once, and starts the fixed comparison only after both A/A screens pass.
It does not change product defaults. Failed screens and assumptions remain
visible in the final report.
