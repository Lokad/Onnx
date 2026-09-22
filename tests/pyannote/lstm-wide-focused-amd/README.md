# Focused numerical qualification of the exact M25 product

Build the full backend test assembly against the unchanged, already-built
candidate Core63d59bad/Data90938e53. Replace only the two project references
with explicit references to the pinned product and its dependencies. Every
source file and fixture remains exact to the prepared candidate, including
the eight new wide-projection cases. Do not rebuild the product.

Six serial AMD jobs: SDK identity, offline restore, backend build, then
focused LSTM tests with AVX512 disabled, ordinary AVX512 and all hardware
intrinsics disabled. Each supported ISA run must pass the complete150-case
original census and all8 new cases. The scalar run must pass140 and retain
exactly the18 previously documented explicit-intrinsics refusals, matching
their names, exception messages and validation stack. Its expected exit1
is declared before execution; any other failure rejects qualification.
No case is omitted or skipped. Test discovery/counters/names are audited.

From repository root use C:/Python313/python.exe -X utf8 -B with test_checks.py,
then run.py prepare,stage,launch,observe,collect and audit.py. New destinations
are artifacts/pyannote-lstm-wide-focused-amd-20260923 and
/dev/shm/lokad-pyannote-lstm-wide-focused-20260923. Existing ones are refused.
All source, dependencies, references and tools are frozen before workers.

These unscored consumer-build/test jobs prospectively use10GiB available
memory and3GiB tmpfs before each worker. This leaves2GiB above the8GiB owned
RSS cap and avoids a12GiB floor tied to retained artifact/cache occupancy.
The monitor still requires1GiB available memory/tmpfs throughout, limits each
job to900seconds, output to1GiB and artifacts to2GiB. CPU2 workers/CPU0monitor,
SDK10.0.204/runtime10.0.8, --tl:off for restore/build, no build servers.
No completed trial's bounds or scored acceptance gate changes. VSTest has
static output and does not accept the MSBuild terminal-logger option.

Exact product hashes are verified in the copied backend output before and
after every suite. Preserve every process/thread birth and resource sample;
collect only terminal owners. Full captured-model replay, dispatch/codegen,
component timing and full application qualification remain separate gates.
