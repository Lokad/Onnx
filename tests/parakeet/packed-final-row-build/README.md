# Build and qualify the proved packed final-row route

Build only the isolated source e3ca64b5. It removes dense reconstruction from
the eligible odd-row path and imports the exact helper proved at f4cc1ea8.
The qualified root and BENCHMARK.md remain unchanged.

The compiled audit allows removal of OwnedRemainderSource, the changed private
RunOwnedPackedRows signature/body, its two callers and one new helper. It proves
the expected compiler-generated ordinal shift without ignoring instructions,
literals, branch offsets or flags. Data methods and public interfaces must stay
unchanged. The new helper must match the standalone proof's compiled body and
AggressiveOptimization flag exactly. Nine negative/positive audit tests pass.

After that audit, execute 41 normal, 41 AVX512-disabled and two actual
hardware-disabled focused contracts. These include actual runtime hashes and
new zero-reconstruction checks at all seven affected row counts. Full model,
copy counters, application and release qualification remain separate work.

Use `C:/Python313/python.exe -X utf8 -B`. Run `test_compiled_scope.py`, then
`run.py prepare`, `stage`, `launch build`, `observe build`, `collect build`,
and `review.py build`. Only after review passes, use `launch capture`,
`observe capture`, `collect capture`, and `review.py capture`. Inspect terminal
PID/birth identity and collect/review each stage once. Preserve failed verdicts.

The build uses the existing SDK10.0.204/runtime10.0.8 and offline package feed
on the exclusive VM. Worker CPU2 and monitor CPU0, 3 GiB owned RSS, 1 GiB free
memory/tmpfs floor and 512 MiB output are bounded prospectively. The two known
Core nullable warnings are retained; no additional warning is admitted.
