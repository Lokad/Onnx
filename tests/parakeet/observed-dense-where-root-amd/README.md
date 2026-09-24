# Pending normal-root qualification

This directory currently contains a draft of 28 focused public Where regression
cases in `DenseScalarWhereTests.cs.txt`. It has not been compiled or integrated.
The cases cover all 19 measured attention lengths, mixed contiguous/outer mask
broadcasts, noncanonical true bytes, exact selected float bits, memory windows,
independent held outputs, higher-rank expansion, invalid shapes and subclasses.
They use independent coordinate/bit expectations and public provider calls.

Finish the normal-root adapter only after the observed-mask candidate passes
the original complete Parakeet admission and all subsequent release gates.
Those gates are fresh shared/e5 and Pyannote correctness, all eight graph cases,
Pyannote timing, both 600-second meetings and 30-second recovery. Use the exact
measured Core f95a13c5 / Data a893952f and source receipt 41f2477c.

The isolated product source has 426 files, versus 425 current root inputs.
Root integration must preserve those exact product bytes and add only this
backend test file, yielding 427 source inputs. Reuse the current release's
normal-root worker and package consumer. Independently compare all 3,253 Core
and 697 Data method bodies, flags and public declarations against the measured
candidate. Keep all existing per-test outcomes in both instruction modes,
then require all 28 new cases. Expected backend totals, if no other source
changes: 3,499 passes / 41 skips normally and 3,409 / 131 with the alternate
instruction mode; both tensor suites remain 368 passes. Do not treat these
prospective totals or this uncompiled draft as qualification evidence.

Build and test only on AMD, using SDK 10.0.204/runtime 10.0.8 and `--tl:off`.
Preserve the original root/package limits and the independent NuGet consumer.
Only the full successful root qualification permits release integration to
remain and BENCHMARK.md to be updated with matching qualified products.
