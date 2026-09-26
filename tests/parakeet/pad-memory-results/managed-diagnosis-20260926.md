# Pad variability includes page faults and scheduling

The unchanged current-root products reproduce substantial copy-path variation
in one four-process diagnostic. Per-call counters support the predicted
page-touching mechanism and also reveal long calls associated with context
switches. This is sufficient to stop searching for another padding kernel;
it does not explain every historical clock or qualify a release.

The exact ORT comparison is now concrete: its retained profile shows no new
live arena space in all 2,880 measured encoder Pad calls. Lokad allocates an
independently owned array on every public call. The same output payload can
therefore encounter a different memory lifecycle in these two contexts.

## What the new counters establish

All means below retain every member of the explicitly named diagnostic group.
Groups describe associations; none replaces an original score or removes a
slow call from qualification.

| Case / process / group | Calls | Mean Pad us | Median Pad us | Median minor faults |
|---|---:|---:|---:|---:|
| attention-51 / candidate 1 / no fault, collection change or switch | 137 | 13.381 | 9.534 | 0 |
| attention-51 / candidate 1 / faults, no collection change or switch | 39 | 71.944 | 71.919 | 41 |
| attention-51 / candidate 2 / all measured calls | 180 | 64.816 | 61.614 | 41 |
| zero-padding / candidate 2 / no fault, collection change or switch | 161 | 35.216 | 36.936 | 0 |
| zero-padding / candidate 2 / faults, no collection change or switch | 13 | 171.723 | 172.550 | 106 |
| zero-padding / candidate 1 / switches, no fault or collection change | 3 | 2,093.019 | 2,093.272 | 0 |

The 166,464-byte attention output contains about 41 pages of data at the VM's
verified 4,096 bytes/page; the 434,176-byte zero-padding output contains 106.
This does not establish alignment or the number of distinct pages in the object.
In candidate 2, all three measured
attention blocks show the fault regime, with no collection-count change or
context switch in any of their 180 calls. Their Pad means are
68.163/62.566/63.719 us. This is sustained variation, not isolated tails.
The output-size/fault-count match supports touching output storage as a material
cost. It does not locate the faulting instruction inside allocation, fill or copy.

The complete zero-padding means are 69.521 and 52.169 us in the two candidate
processes. Candidate 1 has no minor faults in those 180 calls, but three calls
with switches take about 2.09 ms each. Other shapes have both fault-associated
and switch-associated long calls. This rules out treating all variability as
one GC-pause mechanism. The three generic fallback cases remain close between
processes. All original numerical, input and held-output checks pass.

There are unresolved differences even within the no-fault/no-collection-change/
no-switch groups. For convolution-51 those groups average 25.340 versus
12.447 us. Counters do not measure cache state or identify compiled instructions,
and instrumentation changes allocation history. Do not claim that page faults
explain the entire failed uninstrumented screen. That screen remains rejected.

## Instrumentation and scope

The original public Pad timer remains between its two timestamps. Linux thread
resource counters, thread allocation counters and GC counts surround that timer.
The wider bracket includes sampling overhead. Empty-bracket medians are
0.921–0.922 us before the workload and 0.891–0.901 us afterward. Startup-inclusive
means before the workload are 5.152–5.393 us. No overhead is subtracted.

Thread CPU counters often stay unchanged for a short call and then advance by
more than that call's elapsed bracket. Their per-call granularity is insufficient
for exact user/system CPU attribution or subtracting CPU from wall time. Fault
and switch counters are still observations over the wider bracket, not instruction
samples. Record creation happens after the snapshots and can affect later GC.

Both candidate prefixes contain six full rehearsals, versus seven in the
uninstrumented screen; the same fixed elapsed stopping rule is used. Both selected
prefixes contain two. This reinforces that the diagnostic is not a scored retry.
The complete diagnostic covers **187,200 Pad calls, 1,024 empty brackets and
3,120 consecutive 60-call blocks**. All seven jobs are terminal/code zero,
97 files collected, and all 658 resource observations pass; peak owned RSS is
339,644,416 bytes. No product or runtime flag changed, and no native profiler
or additional ORT inference ran.

## Next decision

Keep the already reviewed dispatcher and test it in complete Parakeet execution.
All twelve no-regression gates, numerical/ownership checks and the aggregate
primary repeatability control passed in the original current-root screen; six
individual repeatability controls failed and stay failed. The new counters
establish a material output-memory contribution to short-call variation. They
do not justify another compiler flag, copying variant or warmup-duration search.

The plan changes **eligibility for full-model testing**, not the historical
component verdict: full-model numerical tests may now run despite that failed
component screen. Any promotion must be supported by the unchanged complete
application, per-clip, repeatability, shared-model and ownership requirements,
with the component limitation explicitly retained. An isolated-call stability
claim remains unqualified. No complete application improvement is claimed now.

The source-derived algorithm difference remains the original motivation: ORT
copies contiguous blocks while the qualified Lokad Pad maps each element through
dimensions. The measured padding excess offers roughly a 5.3% application ceiling.
The release stays **53.107381 seconds versus ORT 39.207695 seconds**, and parity
remains outstanding. The next action is to adapt the existing full-model
qualification to the exact current-root pair, reusing the compiled consumers
where compatibility is proven, followed by one complete application comparison.

[All fixed blocks](managed-blocks-20260926.csv),
[every joint cohort](managed-cohorts-20260926.csv),
[empty-bracket calibration](managed-calibrations-20260926.csv),
[identities and all case statistics](managed-observations-20260926.json),
[VM page size and terminal-owner check](managed-platform-20260926.json),
[prospective protocol](../pad-memory-diagnostic-amd/README.md),
[observed ORT allocation](ort-allocation-20260926.md).

Closure: `51e714628218f2908ac46d0e8336b29003d1f7a266e424bd37ec946fa1c7589b`.
Post-collection repository allocation: **47,888,807,264 bytes on disk**;
60,297,909,154 logical bytes. The inventory counts hardlinked paths separately
and excludes directory aliases. Receipt: artifacts/repository-retention-20260923/
pad-memory-diagnostic-closed-allocation-20260926.json. No cleanup is needed to
meet the 50 GB allocated-space target at this point.
