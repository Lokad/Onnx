# Twelve-position weight reuse: AMD qualification

The isolated product changes only Kernel512 from six to twelve spatial positions.
Eight fresh processes exercise original raw, wide-channel raw, added widths21..33,
and all108captured graphs at both AVX2 and AVX512. The spatial family preserves all
original scalar/ordinary graph references and assertions; its Main changes only
the expected Core hash and the width-range starting value. The local compiled
instruction comparison verifies those two changes and all control-flow targets.

From repository root, use C:/Python313/python.exe -X utf8 -B with selftest.py,
then run.py prepare, stage, launch, observe, terminal collect, then audit.py.
Every raw family contains2648cases,20supplementals,10invalids and8004ordinary
requests. All three total24012requests per instruction width; captured layers
add216dispatches and119823360values. Actual width is asserted in each process.

The transport predecessor is terminal current-product profile717686, not a
measurement control. Selected Core remains3c2f; candidate Core is3ca0a2a5.
This lane establishes numerical qualification only, with no performance claim.

CPU2worker/CPU0monitor, runtime10.0.8;12GiBavailable/3GiBtmpfs preflight,
8GiBRSS,1GiBminimumavailable/tmpfs,1GiBoutput/2GiBartifacts and900seconds/worker.
AVX2 uses DOTNET_EnableAVX512=0; AVX512 has ordinary settings. Refuse artifact
reuse; preserve every owner/sample and freeze source/tools before deployment.
