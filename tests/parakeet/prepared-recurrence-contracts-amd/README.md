# Qualify prepared recurrence and retained packing behavior

Consume the unchanged actual M64 product, Core `3c23b44a` / Data `cc37b19e`, after
compiled review `1b7f8e21`. The first build wrapper's refusal `b2d46d3b` is preserved;
the successor reconciled exact compiler identifier changes without rebuilding.
Source `b52a89c1` remains isolated. Root product and BENCHMARK stay selected.

Build only a test assembly with the qualified product DLLs as direct references,
canonical SDK10.0.204 and existing offline xUnit packages. Include all 22 new
source contracts, the retained CPU LSTM, projection, panel admission/overflow and
matrix/aggregate packing contracts. Freeze the full method/case census in
stage.json before execution; require every case executed/passed without skips in
both normal and AVX512-disabled mode. Immutable source hashes prevent changing
that census after observing results.

A separate public-API-only budget anchor uses the same test DLL with the selected
Core `672e5f30` / Data `065b7a7f`: it must fail specifically because the original
product retains zero rather than 13,107,200 prepared recurrent bytes. Its identity
test must pass. The candidate must pass that same anchor along with all other
cases. Verify loaded product/consumer hashes, exact runtime paths, CPU2 affinity,
one processor, runtime10.0.8 and requested instruction mode inside the test host.

Six jobs: SDK version, tests restore, tests build, selected negative, candidate
normal, candidate AVX512-disabled. Namespace
`/dev/shm/lokad-parakeet-prepared-recurrence-contracts-20260924`.
Use CPU2 workers, CPU0 monitor, all .NET commands with `--tl:off --nologo -v minimal`,
no model/download/product rebuild. Prospective bounds: 8 GiB available memory and
2 GiB free tmpfs at preflight, RSS below 8 GiB, at least 1 GiB available memory and
tmpfs during each job, 1 GiB output/job, 2 GiB total artifacts, 900 seconds/job and
four hours total. Full-model/application bounds are unchanged.

From repository root run `C:/Python313/python.exe -X utf8 -B` with this directory's
`run.py prepare`, `stage`, `launch`, `observe` while active, then `collect` only
after terminal PID/birth owners. Run `audit.py` to independently reconcile every
test result, loaded identity and resource sample. Preserve failures and correct
harness faults in successors; never edit frozen sources or overwrite hardlinks.
No component timing or release admission follows from these tests alone.
