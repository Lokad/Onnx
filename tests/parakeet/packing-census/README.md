# Prepared Parakeet weight census

This diagnostic consumer loads the exact qualified production assemblies and
local model assets, creates Memory-policy contexts, and records actual prepared
weight mappings. It performs no inference. The auditor matches every graph
descriptor to the complete qualified trace and joins mapping presence to node
cost; it does not infer which kernel executed from mapping presence alone.

The [completed report](../performance-profile/results-20260921.md) contains
counts, costs, source comparisons, limits and evidence identities.

Original commands, from the repository root:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/packing-census/run.py
    C:/Python313/python.exe -X utf8 -B tests/parakeet/packing-census/audit.py

They refuse the already-closed artifact. The worker uses CPU 2 before CLR
startup, normal runtime settings, a 10 GiB available-memory preflight, an 8 GiB
RSS ceiling, 180 seconds and 20 GiB free disk. Only its owned child can be
terminated. Product code and pending AMD campaigns remain unchanged.
