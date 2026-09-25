# Observe the direct candidate without changing its arithmetic

Use the qualified `direct-depthwise-source-v2` snapshot. Preserve every helper
instruction and option/dispatch condition. Add only the established observer
calls in `Conv2DFloatCore`, `RunTiledBatchFloat` and `RunFloatMatMulKernel`.
The successful direct return now records completion, while all old patch/view/
matrix sites remain observed and must stay at zero for the targeted geometries.
Source insertions reverse exactly. The private observer adds 16 methods.

The one-time preparation requires completed focused numerical qualification:

    C:/Python313/python.exe -X utf8 -B tests/parakeet/direct-depthwise-observer-source/prepare.py

Run the three scope tests first. No model or build runs on Windows.
