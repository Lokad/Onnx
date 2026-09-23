# Finite extremes across convolution channel blocks

Build only the new ConvExtremes driver, referencing current public Core APIs.
Reuse the exact qualified M27 raw consumer (bffad772) by reflection; its
GraphRaw.Create/Run/Check methods are unchanged from the original qualification.
The driver calls the ordinary prepared graph path. No product, kernel or
qualified graph caller is rebuilt.

Four fresh processes compare current and candidate with AVX512 disabled and
enabled. Each has 1280 cases: c64/80/128/256, m32/48, height3, widths7/13,
strides1/2, five finite patterns and all eight bias/residual/ReLU combinations.
Patterns include extreme finite magnitudes, subnormals, signed zeros, overflow
and cancellation across channel blocks. Operands are explicitly finite.
Each case reuses the qualified graph check for one unprepared control and two
prepared executions, including scratch, readonly operands and held results.
Two additional prepared executions record exact output hashes and ownership.
Thus each process has 6400 graph calls. The independent auditor requires every
case and exact current/candidate hashes and graph records within each width.
No sample is timed and no speed or application claim follows.

Three SDK/restore/build jobs precede the four numerical workers. CPU2 workers,
CPU0 monitor, SDK10.0.204/runtime10.0.8, original global.json and offline feed.
All restore/build commands use --tl:off --nologo -v minimal. Keep12 GiB available
and3 GiB tmpfs preflight,8 GiB owned RSS,1 GiB live available/tmpfs/output,
900 seconds/job and2 GiB artifacts. Existing numerical closure c2be39dc and
build closure b8573a2d are mandatory prerequisites.

From repository root use C:/Python313/python.exe -X utf8 -B with run.py
prepare,stage,launch,observe,collect then audit.py. New artifact:
artifacts/pyannote-convolution-channel-extremes-amd-v2-20260923; new VM directory:
/dev/shm/lokad-pyannote-convolution-channel-extremes-v2-20260923. Refuse existing
namespaces, preserve failures and collect only terminal owners. Never observe
after closure. Actual code inspection and fixed complete-call timing remain next.
