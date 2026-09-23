# Completed M42 diagnostic

See [the report and complete clocks](../dispatch-events-results/report-20260923.md).
No product was admitted and no benchmark measurement was replaced.

The original frozen sources and outputs remain unchanged. `audit_evented.py`
corrects the initial auditor's assumed Speedscope schema; the bundled exporter
produces evented profiles. Ten boundary-accounting and seven evented-stack tests
pass. Capture closure: `c6e1e4d`.

`full_export.py` builds `ExportAll.cs.txt` as an offline reader on AMD. It exports
every record from the already closed traces, including raw payloads for unknown
events, then verifies the complete provider census and all original typed/custom
records. Closure: `35b18e87`. Its initial extraction needed
`full_export_stage_resume.py` to invalidate Python's importer cache before using
newly extracted modules. This happened before payload creation or workers.

All capture/export workers are terminal. Existing namespaces intentionally
refuse reruns. `publish.py` derives diagnostic CSV/JSON evidence without scoring.
The next optimization must use a distinct source candidate and new qualification.
