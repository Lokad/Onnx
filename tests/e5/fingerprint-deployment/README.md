# Independent e5 cache deployment and fresh native baseline

This experiment compares fresh sequential processes for both public Default
and Memory policies. Disabled graphs never construct the fingerprint cache;
enabled graphs use the ordinary environment switch from startup. Native ORT
1.23.2 executes in its own process with one thread and the same CPU and inputs.
No inference processes overlap or share a graph, cache or weight allocation.

The preceding [common-state comparison](../fingerprint-balanced/comparison-results-20260920.md)
passed its fixed screens, with a resident cache for every label. This separate
experiment measures ordinary deployment and supplies new native timings.
The cache remains off by default while qualification is pending.

Each phase has 160 workers: four visits, five cases, two policies and four roles.
A/B are identical disabled controls; C is disabled in A/A and enabled in the
conditional comparison; N is ORT. Four balanced orders put every role once in
each position per case/policy. Case direction and policy order alternate.
`protocol.py` fixes every job before data collection.

Each process retains model loading and its first request, conditions for thirty
cumulative Execute/Run seconds, then measures 48 blocks with 32/16/4/4/2 calls
for 8/30/padded128/128/512 tokens: 89,088 measured calls per phase. Each call
records public Execute/Run and enclosing Reset/disposal-plus-Execute/Run.
All samples, allocations and GC tails remain. There is no forced GC, profiler,
runtime override, adaptive measured count or sample exclusion.

For every case/policy and both boundaries, duplicate controls must stay within
0.5% in aggregate, 1% in each visit and matched position, and position contrast
at most 1.01. Controls compare all three managed pairs in A/A, and B/A in the
candidate phase. Only a completely passing A/A receipt enables that already
frozen comparison. Candidate/control must be at most .98 at 8 tokens, .99 at 30
and 1.01 elsewhere, with every visit at most 1.02. These are descriptive screens;
per-call observations are not independent process repetitions or confidence
bounds. Fresh ORT ratios do not calibrate managed controls.

The qualified archived core is `faf2844`, SHA256
`48ca1d62ee2586d81072b8e347a671013d00abf8fe637c89eff65314e13cc710`.
The probe is a source-pinned local build. Actual CPU2 affinity is inherited
before CLR startup; AMD uses .NET 10.0.8 and AVX-512. Complete before/after
arrays must pass native scaled error <=1e-4, managed bytes must match across
settings, and inputs, held outputs, graph fingerprints and cache state must
remain unchanged. Managed workers must not load native ORT.

Workers are limited to 300 seconds, 6 GiB group RSS and 1 GiB available memory;
observed foreign CPU must be <=2% of machine capacity and guest steal <=0.5%.
Snapshots miss some short-lived work and do not observe hypervisor neighbors.
All process births must terminate before collection. Collection streams the
archive locally to avoid duplicating large results on the small VM disk.

Build `Probe.csproj` in Release with `--tl:off --nologo -v minimal`, setting
`FrozenProductDirectory` to the archive-qualified product replay's Release
directory and `FrozenOrtDirectory` to `artifacts/e5-public-ort-20260919/bin`.
Use artifact `artifacts/e5-fingerprint-deployment-20260920`. Run `smoke.py` and
`test_audit.py` before committing and freezing with `prepare.py`. Then use
`vm.py launch|poll|collect --artifact <artifact> --phase aa|compare`, `audit.py`,
`close_phase.py` and `finalize.py` in that order as specified in the prospective
plan `.agent/m2-fingerprint-deployment-20260920.md`.

Successful writers refuse existing outputs. A tool observation timeout never
authorizes restarting a still-live process. Preserve failed controls and all
earlier evidence. No product code, default or dependency changes in this lane.
