# Normal AMD build of the isolated Winograd product

M33 admitted component closure 302de6c9 permits this build. Frozen source
8814947c copies the exact two qualified arithmetic files and changes packing,
stride-one dispatch and focused tests only; every qualified root file remains.

Four jobs identify SDK10.0.204, restore/build normally and inventory every Core
and Data method. Public declarations and Data methods must remain exact.
Sixteen new methods implement Winograd, optional weights and dispatch; only
specified graph methods, dispatch and generated record members may change.
The complete inventory retains all instruction bodies for independent review.
No numerical or performance result is inferred from compilation.

Use run.py prepare,stage,launch,observe,terminal-only collect, then audit.py.
Worker CPU2, supervisor CPU0, preflight10GiB available/3GiB tmpfs, live8GiB RSS
ceiling and1GiB available/tmpfs floors,900seconds perjob,1GiB output/2GiB
artifact ceilings. No shared build servers, profiler or JIT overrides. Every
input, dependency and runtime is frozen; preserve all failed attempts.
