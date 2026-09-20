# Interleaved independent e5 processes

This prospective experiment reduces temporal separation between engines while
keeping their runtimes, collectors, graphs and weights in separate processes.
Inactive processes are suspended; only one engine executes at a time on CPU2.
The supervisor uses CPU0. Physical caches and memory remain shared, so the
result must identify its resident-process boundary and report solo bridges.
It does not silently replace the earlier isolated-deployment measurements.

The [complete old-data diagnostic](../deployment-variance/results-20260920.md)
retains every call and original failed verdict. Persistent mean/median offsets
and within-worker drift motivate this distinct scheduling experiment, without
claiming their cause. The prior shared-collector paired experiment also remains
failed. No product or default changes are part of this lane.

Each frozen phase has 40 cohorts / 160 fresh workers: four visits, all five
e5 cases, both public Default/Memory policies, roles A/B/C/N. A/B use normal
defaults; C is identical in A/A and enables fingerprint strings plus wider
LayerNorm only in the conditional comparison. N is Microsoft ORT1.23.2, one
intra/inter-op thread, sequential, all optimizations, no spinning. The current
archive-qualified core is4f10e8b, SHA256
`8b991fd7baaa470c45285754b20696c463dedc890a7db23dd4f0b9c7c818ccf1`.

Every worker keeps its actual first output, then conditions to both 128 calls
and 30 cumulative Execute/Run seconds. Conditioning stops at the first call
satisfying both conditions. Every first/conditioning call remains. Forty-eight
measured cycles contain all four roles; each 24-cycle block contains every role
permutation exactly once, deterministically shuffled before observations.
Batches contain32/16/4/4/2calls for8/30/padded128/128/512tokens, totaling89,088
measured calls per phase. The first-created worker measures64solo calls before
others load; the last measures64solo calls after all others are actually
terminal. Four balanced creation orders give every role both solo positions
once per case/policy, adding5,120solo calls per phase.

Anonymous pipes carry fixed eight-byte command/acknowledgement packets outside
timing. Worker storage is preallocated; no per-batch JSON or growing timing
lists are used. Every request records public Execute/Run and the enclosing
Reset/disposal-plus-execution, allocation and all GC-count deltas. There is no
forced GC, profiling, runtime override, adaptive measured count or exclusion.
Native/managed library identities, actual switches, complete before/after
arrays, first-output/input preservation, fingerprints and cache state are checked.

Controls retain the original empirical limits for both boundaries and every
case/policy: aggregate managed ratios within0.5%, visits and pooled positions
within1%, and largest/smallest position ratio<=1.01. Every observed resident/solo
ratio must lie[.95,1.05], including native. A/A compares all three managed pairs;
the candidate phase checks B/A. Candidate C/mean(A,B) must be<=.98at8tokens,
<=.99at30 and<=1.01elsewhere, with every visit<=1.02. These are screens, not
confidence intervals or independent per-call observations. A completely passing
independently audited A/A receipt is required before the frozen comparison.

Each cohort has600seconds,12GiB combined worker RSS and1GiB minimum available
memory. Process birth, affinity, suspension state and inactive CPU counters
are checked; every resource sample is retained. Observed foreign CPU must stay
<=2%ofmachinecapacity and gueststeal<=0.5%. These snapshots miss some exited
work and do not observe hypervisor neighbors. Solo order/time can still confound
bridges, and physical-cache effects are not eliminated by process isolation.

Build `Probe.csproj` in Release with `--tl:off --nologo -v minimal`, setting
`FrozenProductDirectory` to the archive-qualified LayerNorm replay's Release
directory and `FrozenOrtDirectory` to `artifacts/e5-public-ort-20260919/bin`.
The probe is a source-pinned local build; its exact sources and binaries are
frozen together. It is not described as a product source-archive build.

Run `smoke.py --artifact <new-directory>` locally. It executes80actual worker
smokes: all cases/policies, both settings, four private engines, every command,
both solo positions and full array/ownership checks. It supplies no timing
verdict. Run `test_audit.py --artifact <directory>` for real-record refusals,
schedule balance, synthetic drift/bias and resource/command faults. Preserve
all failed attempts. Windows workers are detached with explicit pipes to avoid
an extra console-host process; actual timing is Linux-only.

After committing sources, `prepare.py --artifact <directory>` freezes both
phases and all hashes before VM timing. `vm.py launch|poll|collect --artifact
<directory> --phase aa|compare` uses the designated exclusive VM. Collection
streams locally to avoid a duplicate large remote archive. Audit the collected
payload with `audit.py --payload <collected-phase> --phase <phase> --output
<new-file>`, then close evidence. Successful writers refuse existing outputs;
an observation timeout never authorizes restarting a live job.

Current staging directory is `artifacts/e5-interleaved-processes-v3-20260920`.
The first failed local attempt and a bounded console-host diagnosis remain in
`e5-interleaved-processes-20260920`; the second local attempt remains in `-v2-`.
Neither failed attempt contains accepted timing evidence. The focused plan is
`.agent/m2-interleaved-processes-20260920.md`.
