# Parakeet dispatch runtime events

Diagnostic only: current/rejected-M41 actual DLLs, same21fixtures/120calls,
all begin/end markers, complete CLR events and sample stacks. Normal tiering,
no forced GC, no admission score or replacement timing. M40/M41 stay rejected.
Consumer and exporter build on AMD SDK10.0.204 from pinned local references.

Use C:/Python313/python.exe -X utf8 -B with run.py prepare, stage, launch,
observe, collect, then audit.py. Each namespace is exclusive. Stage extraction
and remote preparation occur in separate SSH processes; guard limits unchanged.
