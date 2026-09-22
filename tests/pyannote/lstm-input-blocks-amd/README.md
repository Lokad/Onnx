# Actual AMD LSTM qualification

This campaign copies the locally qualified four-row projection product and
selected control without rebuilding either. The fixed nine workers run150focused
tests at AVX2/AVX512, all captured recurrent outputs on both products at both
instruction settings and with hardware intrinsics disabled, and fresh ORT1.29
executions of all twelve original LSTM graphs. Complete tensors, scratch,
ownership, input identities, CPU2 affinity and terminal process identities are
checked. System.Numerics vector width is recorded independently of AVX512 support.

From repository root, use `C:/Python313/python.exe -X utf8 -B` followed by:

    tests/pyannote/lstm-input-blocks-amd/selftest.py
    tests/pyannote/lstm-input-blocks-amd/run.py prepare
    tests/pyannote/lstm-input-blocks-amd/run.py stage
    tests/pyannote/lstm-input-blocks-amd/run.py launch
    tests/pyannote/lstm-input-blocks-amd/run.py observe
    tests/pyannote/lstm-input-blocks-amd/run.py collect
    tests/pyannote/lstm-input-blocks-amd/audit.py

Prepare pins the actual installed SDK, runtime and imported native/Python
dependencies before deployment. Existing destinations are refused. Observe until
all owners are terminal; collect retains failures too. No retry changes a frozen
input. Limits remain12GiBavailable/3GiBtmpfs before each worker,8GiBowned RSS,
1GiBminimum available/tmpfs,1GiBjob output,2GiBartifact size and900seconds per
worker. Monitoring runs onCPU0; children inheritCPU2 before CLR/native startup.

The full hardware-disabled xUnit run is deliberately absent:18existing tests
explicitly request unavailable FMA. Local evidence retains their exact refusals,
132supported passes and36new passes. AMD scalar replay uses valid Auto options.
All150tests must pass in both supported AMD instruction settings.

These are numerical qualifications. They produce no timing score or application
parity claim. Generated code and a prospectively fixed complete-call screen follow.
