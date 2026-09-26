# Observe rejected padding in the complete application

This diagnostic asks whether bypassing encoder PadCore changes compilation of
the real frontend reflection fallback. It reuses selected Core 521bae17 and
rejected M47 Core 060950a1; neither is the current release. Their original Data
methods are identical to the retained Data observer a2a0b490. No product,
observer, kernel, runtime setting, warmup count or performance gate is changed.
Reuse the existing gzip event exporter, whose decompressed records were already
verified against the original exporter. Compression keeps complete raw events
within the storage bounds; it does not filter or summarize the event stream.

One process per binary runs the original 20 clips in order, with one warmup and
three measured passes. Both use the same wall observer. EventPipe attaches
before model construction and the first warmup. Every request has paired
markers; additional clock anchors bound conversion between wall timestamps and
runtime event time. Retain all 160 requests, frontend reflection and 48 encoder
pads per request, compilation/method-load events, allocation/GC data, complete
native/public results, immutable inputs and independently owned held outputs.

Build only an adapted SampledAudio consumer and the existing compiled-method
inspector on the AMD VM. Preserve the 164 original consumer methods except
Main. Main changes only startup waiting and markers outside request timers.
Resolve every operand in both consumers and Data assemblies against each exact
Core before inference. Reuse the original numerical validator and phase/node
interval auditor byte for byte. The original failed screen remains rejected.

Fix limits before preparation: CPU 2 computes; CPU 0 supervises, collects and
exports. Require 11 GiB available RAM and 2 GiB free tmpfs before each job,
waiting at most 900 seconds for RAM. Bound owned RSS at 12 GiB, remaining RAM
and tmpfs at 1 GiB, output at 512 MiB/job and 1 GiB total, 900 seconds/job and
four hours overall. Collect only after all recorded process identities terminate.
Require zero lost events, every marker and clock anchor, every original output
check and all monitored affinity/resource checks. Never overwrite a namespace.

Report per-request fallback time and available compiled versions throughout all
passes, encoder padding and total request clocks, plus JIT/GC overlaps. A loaded
version is not proof that a particular invocation executed that version. Keep
clock-conversion uncertainty explicit. Both processes are instrumented; there
is no uninstrumented control, admission score or fresh ORT comparison. This can
identify a runtime-history explanation or remain inconclusive; it cannot admit
M47 or justify changing thresholds. A future implementation needs a distinct
causal rationale and all original correctness and performance gates.

Run `C:/Python313/python.exe -X utf8 -B run.py prepare`, then `stage`, `launch`,
`observe`, `collect` separately from this directory. Preparation freezes every
tool and source; review and complete the auditor before preparation. Run
`audit.py` once after collection. Preserve a failed run without retrying it.
