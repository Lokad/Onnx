# Runtime diagnosis of the retained e5 regression

Four processes run selected, candidate, candidate, selected using the exact
M66 product binaries and 30-token e5 inputs. Every original warmed-consumer
statement remains; four reversible hooks observe setup and call boundaries.
The original result schema, 780 calls, 600 warmup labels, graph execution,
native bounds, hashes and ownership checks are unchanged.

CPU 2 executes, CPU 0 collects full compilation/GC events and later exports
every raw event. Require 6,240 markers, zero lost events and all output hashes
equal to the original comparison. No runtime override, forced GC, product
rebuild, admission score or native timing. Instrumentation can change runtime
history; associations cannot retroactively prove the old failure's cause.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`,
`launch`, `observe`, `collect`, then `audit.py` from this directory. Preparation
verifies the failed graph closure and exact original source recovery. All .NET
work runs on the authorized AMD VM with SDK 10.0.204 and runtime 10.0.8.

Prospective bounds: 11 GiB available RAM/3 GiB tmpfs preflight; 8 GiB owned RSS,
1 GiB minimum remaining RAM/tmpfs, 256 MiB output/job, 512 MiB stage,
900 seconds/job, four hours total. All processes must terminate before
collection. Preserve failed stages and the original graph failure `729814e9`.
