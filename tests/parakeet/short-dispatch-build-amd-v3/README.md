# Build the non-inlined general and short Parakeet dispatchers

Source918a28ab adds NoInlining to both new helpers. The prior layout passed
numerics but generated a4092byte wrapper containing the general dispatcher;
no timing was run. This fresh normal build retains the strict method inventory
and exact original general IL, with only the verified static-initializer rename.

Use C:/Python313/python.exe -X utf8 -B with run.py prepare, stage, launch,
observe, collect, then audit.py. SDK10.0.204, actual Core/Data, CPU2. Root420
inputs remain unchanged. Numerical/code-generation and performance admission
follow separately. The same fixed21fixtures and all original gates apply.
