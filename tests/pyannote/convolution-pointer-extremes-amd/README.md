# M28 finite-extreme qualification with the exact existing driver

Reuse ConvExtremes2d7c8ad1 from closed M27a2c99e95, without rebuilding it.
Both roles load the new numerically qualified raw GraphRaw consumer. Only
the selected/candidate product and declared probe hash differ at invocation.
Four fresh selected-256/candidate-256/selected-512/candidate-512 processes
retain1280cases/6400graphcalls each: channels64/80/128/256,m32/48,h3,w7/13,
strides1/2,five finite patterns and all eight epilogues. Exact output hashes,
graph records, scratch, readonly inputs and retained-output ownership are
mandatory. No timing or speed claim. Explicit AVX512-disable flag for256.

CPU2workers/CPU0monitor,.NET10.0.8. No SDK/restore/build jobs. Original limits:
12GiBavailable/3GiBtmpfs preflight,8GiBRSS,1GiBlive available/tmpfs/output,
900seconds/job,2GiBartifacts. Freeze prior numerical/build/driver closures.
Python3.13 -X utf8 -B run.py prepare,stage,launch,observe,collect then audit.py.
Fresh artifact artifacts/pyannote-convolution-pointer-extremes-amd-20260923;
VM /dev/shm/lokad-pyannote-convolution-pointer-extremes-20260923.
Preserve failures, collect only terminal owners and never observe after closure.
