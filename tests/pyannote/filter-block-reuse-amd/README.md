# Four-block convolution: actual AMD numerical qualification

Reuse the exact locally qualified Core `64f42e97` and three consumers; no rebuild.
Six fresh CPU2 workers run original raw, wide raw and all captured layers in
both AVX2 and AVX512. AVX2 uses `DOTNET_EnableAVX512=0`; AVX512 uses ordinary
runtime settings. Each consumer asserts its actual instruction width.

Each raw family retains all 2,648 cases, twenty supplementals, ten invalid/alias
checks and 8,004 ordinary graph requests. The wide family exercises 64/128 output
channels and 64-channel supplementals. Every inherited numerical, scratch,
prepared-weight and ownership assertion remains. Both layer workers validate
108 graphs, 216 calls and 119,823,360 values against prior output and native arrays.

CPU0 supervises. Original bounds remain: 12 GiB available / 3 GiB tmpfs preflight,
8 GiB owned RSS, 1 GiB minimum available/tmpfs, 1 GiB output, 2 GiB artifacts,
900 seconds per worker and four hours overall. All process/thread affinities,
identities and resource samples are retained. Collect only terminal owners.

From root with `C:/Python313/python.exe -X utf8 -B`:

    tests/pyannote/filter-block-reuse-amd/run.py prepare
    tests/pyannote/filter-block-reuse-amd/run.py stage
    tests/pyannote/filter-block-reuse-amd/run.py launch
    tests/pyannote/filter-block-reuse-amd/run.py observe
    tests/pyannote/filter-block-reuse-amd/run.py collect
    tests/pyannote/filter-block-reuse-amd/audit.py

Existing destinations are refused. Executed inputs and tools are immutable;
retain failures and correct with explicit successors. The old AMD fixture
directory remains a pinned external dependency. No source integration,
performance score or generated-code claim follows from this qualification.
