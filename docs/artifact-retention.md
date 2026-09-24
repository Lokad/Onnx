# Local artifact retention

The 2026-09-23 cleanup removed **119.94 GB across 210,772 obsolete artifact
files**. The repository then contained **37.15 GB**, including 23.74 GB in
`artifacts` and 10.79 GB in canonical `models`. Counting the worktree's junction
to those same models again gives **47.94 GB**. Both measurements are below the
requested 50 GB limit; units here are decimal GB.

The latest inventory is **48.51 GB**, including **35.01 GB** in artifacts.
It retains the selected release, rejected trials, exact ORT diagnosis, matched
Lokad profiles, both independent application admissions and the combined-source
build qualification. The first combined tensor run's source-policy failure and
the corrected two-mode qualification remain recorded. Complete native/public
Parakeet models, the combined application, shared/e5 and complete Pyannote
correctness are collected and audited. The graph comparison, its retained e5
failure, the runtime diagnostic and the admitted focused e5 successor are all
retained. Complete Pyannote timing/meetings and normal root/package qualification
have passed. The full Parakeet masking/padding layout capture is closed and
published; the next isolated composition reuses the qualified dense-mask helper.
Its build, compiled review and complete operator numerical census have passed.
The first inspection's missing-dependency failure remains recorded. No new
model copies were needed.

All raw traces, clocks, native arrays, failed checks and qualification evidence
remain retained. The inventory totals **48,509,877,335 bytes**, with
**1.49 GB** of headroom. Receipt:
`artifacts/repository-retention-20260923/m70-numerics-closed-size.json`.
Counting the model junction again gives59.30GB, which counts the same10.79GB
of canonical models twice. The inventory excludes that directory alias and
counts each actual file path, including hardlinked paths separately.

Before the observed-mask numerical checks, **45.42 MB across 184 closed VM
files** was retired: generated build packages and a duplicate instruction export.
Seven original package archives and the complete local export remain verified;
every immutable payload dependency stayed intact. Receipt:
`artifacts/parakeet-observed-where-build-retention-20260924`.
A separate layout-cache retirement refused before deletion because that cache
was already empty. Its read-only review is retained in
`artifacts/parakeet-masking-layout-build-cache-retention-refusal-20260924`.

The new decoder capture stores all 380 complete LSTM calls in only **27.37 MB**
of unique tensors by deduplicating constants and repeated values. It reuses the
canonical decoder and saved encoder outputs. No model copy or download was
needed. The compiler-name review reused the original binary and instruction
inventory without another build. Compact tracked reports refer to the retained
raw artifacts. No further deletion is necessary to meet the current 50 GB cap.

The ORT diagnosis additionally retired **182.21 MB** of verified VM-only profile
duplicates after collection; complete local traces and their archive remain.
Receipt: `artifacts/parakeet-ort-remote-profile-retention-20260924`. Optimized
graph serialization retained node metadata and identities, then removed its
declared scratch weight sidecars after closing each session, including the
2.476 GB encoder sidecar. No second model-weight copy was transferred locally.

Before combined-candidate Pyannote qualification, a memory-preflight refusal
was recovered by retiring **141.10 MB**: the closed VM `perf.data` duplicate.
The complete local raw trace, collection and archive remain hash-verified.
All VM payload/stage dependencies were checked and every recorded owner was
terminal. No inference overlapped this retirement or was repeated; the original
11 GiB preflight threshold remains unchanged. Receipt:
`artifacts/parakeet-native-perf-remote-retention-20260924`.

Before the prepared-recurrence full-model checks, another **558.69 MB of tmpfs**
was recovered by sharing byte-identical files in terminal VM campaigns. The first
pass replaced 1,449 binary/metadata duplicates and verified 10,141 paths; the
second replaced 21,763 source/tensor duplicates and verified 49,749 paths. All
contents, permissions and paths stayed intact, with PID/birth owners checked
terminal before replacement. Receipts:
`artifacts/parakeet-prepared-recurrence-closed-links-20260924` and
`artifacts/parakeet-prepared-recurrence-input-links-20260924`.

Before the selected-release profile, another **195.49 MB** in eight closed VM
output duplicates was retired after verifying the complete local control proof
and every VM payload dependency. All local clocks, result files and collection
archives remain. Receipt: `artifacts/parakeet-balanced-control-remote-retention-20260924`.
This maintenance followed a memory-preflight refusal before any profiling worker;
the original failure and recovery are retained with unchanged inputs and limits.

After that profile closed, four duplicate VM trace exports totaling **279.62 MB**
were retired before the inclusive-packing build. Complete local Speedscope and
Chromium exports, their archive and raw traces remain hash-verified. Every
capture/export owner was terminal and immutable VM dependencies were protected.
Receipt: `artifacts/parakeet-selected-profile-remote-retention-20260924`.
A transport import failure occurred during the initial read-only check, before
deletion; its record is retained in the adjacent `-transport-refusal` directory.

Before the focused packing contracts, another **132.35 MB** in four closed VM
duplicates was retired: two instruction inventories and two older startup
traces. Complete local inventories, raw traces and archives remain verified,
and all immutable VM inputs are protected. Receipt:
`artifacts/parakeet-inclusive-packing-precontract-retention-20260924`.

After the product build and focused contracts closed, **85.58 MB across 542
generated package-cache files** was removed from the VM. All 20 package archives
matched retained feed originals, and all immutable inputs and products remained
exact. This recovered **86.81 MB of tmpfs** before model residency work. Receipt:
`artifacts/parakeet-inclusive-packing-package-retention-20260924`.

Current release evidence, current Parakeet experiments, explicit dependencies
of the active benchmark tools, source files, reports, clocks, audio assets,
canonical models and worktrees were retained. Removed files were older duplicate
models, intermediate tensor dumps, binaries, build packages and transport or
backup archives. Historical campaigns may no longer replay from their original
binary paths; their retained reports do not imply that every old binary remains.

After deletion, all **12 selected release/Parakeet proof records and 4,602
referenced files** matched their hashes. All **420 selected root product inputs**
also matched. This includes the current release comparisons, root/package
qualification, original Parakeet matrix capture, rejected M41 qualification and
the complete M42 trace evidence. Product source and `BENCHMARK.md` were unchanged.

The ignored local directory `artifacts/repository-retention-20260923` contains
the reviewed `proposed.json`, per-file SHA-256 retirement journals,
`completed.json`, `verified.json` and `junction-accounting.json`. Deletion was
restricted to listed regular files within this repository's `artifacts` folder;
tracked files, changed files, reparse-point paths and protected directories were
refused. A bounded Python controller resumed the same hash journal after the
PowerShell controller proved slow across many small files. This local cleanup
used no recursive removal.

Subsequent VM maintenance retired verified, locally retained copies of closed
diagnostic exports and build caches. Identical runtime binaries were hardlinked
without changing their paths or bytes. These separate receipts are under
`artifacts/parakeet-first-use-kernels-*-retention-20260923`,
`artifacts/parakeet-first-use-kernels-build-maintenance-20260923` and
`artifacts/parakeet-first-use-kernels-runtime-links-20260923`.

For subsequent experiments, reuse retained inputs, avoid whole-model copies,
and check the total repository size before preparing large bundles. A new cleanup
requires a fresh dependency review and an explicit manifest; the one-time scripts
in `eng` are not an automatic age-based deletion policy.

Run `C:/Python313/python.exe -X utf8 -B eng/measure-repository-size.py` for a
read-only fresh inventory. Add `--output artifacts/<new-receipt>.json` to retain
it in an existing directory; the command refuses an existing output file.
It sums regular file lengths, counting hardlinked paths separately, and reports
directory aliases without traversing them. This is logical size, not allocated
filesystem space. Known aliases to top-level directories also get an explicit
duplicate-count total. The inventory never deletes files.

M50 VM maintenance additionally retired 269.17 MB: 17.94 MB of regenerable
build packages and 251.23 MB of duplicate inventory/event exports. All removed
exports have verified complete local copies, all owners were terminal, and
every immutable VM payload path was protected. No model or raw trace was removed.
Receipts are in artifacts/parakeet-isolated-short-kernels-cache-retention-20260923,
artifacts/parakeet-redundant-padding-export-retention-20260923 and
artifacts/parakeet-redundant-matrix-inventory-retention-20260923.

M52 VM maintenance retired another 17.94 MB of generated build packages.
Seven archives matched the protected offline feed; all immutable payloads and
current runtime files remained intact. Receipt:
`artifacts/parakeet-wide-projection-cache-retention-20260923`.

M54 VM maintenance retired 17.94 MB of generated build packages and 111.88 MB of
six duplicate inventory/event exports, each retained and verified locally.
All raw traces and immutable inputs remain. It also replaced 284 identical
runtime binaries with hardlinks after verifying 1,784 file hashes and every
terminal owner, freeing 77.55 MB of tmpfs without changing file bytes or paths.
Receipts: artifacts/parakeet-wide-entry-cache-retention-20260923,
artifacts/parakeet-redundant-wide-entry-export-retention-20260923 and
artifacts/parakeet-wide-entry-runtime-links-20260923. The application timing
run starts after this maintenance; no cleanup overlaps its measurements.

During subsequent Pyannote correctness qualification, a memory preflight waited
between completed workers. The supervisor was suspended only after verifying
its exact PID/birth, completed-worker state and absence of children. Linking
1,822 identical files in closed campaigns, excluding the active namespace,
freed another **219.03 MB of tmpfs**. All 14,298 hashes and active inputs were
verified, then the same supervisor resumed. No inference was stopped or repeated;
the original memory bound stayed intact. Receipt:
`artifacts/parakeet-wide-entry-pyannote-preflight-maintenance-20260923`.

After all M54 qualification workers finished, idle VM deduplication linked
1,275 identical files after verifying 16,059 hashes, freeing **180.28 MB of
tmpfs**. All recorded owners were terminal and file bytes remained exact.
Receipt: `artifacts/parakeet-wide-entry-shared-vm-maintenance-20260923`.

Before the next build, idle retirement removed 1,243 closed-root cache and
duplicate inspection files, totaling **243.53 MB** and freeing **185.32 MB of
tmpfs**. Every removed export has a verified local copy; all 47 package archives,
immutable inputs, runtimes and source remain. Receipt:
`artifacts/parakeet-wide-entry-root-retention-20260923`.

After the ordered-block build closed, retirement removed its generated package
cache and duplicate instruction export: **45.20 MB across 184 files**, freeing
45.57 MB of tmpfs. All seven package archives and the complete local export
remain verified. Receipt:
`artifacts/parakeet-ordered-wide-blocks-build-retention-20260923`.

After numerical qualification, 44 identical runtime binaries were linked after
verifying 129 hashes, freeing 15.66 MB of tmpfs. The subsequent screen's memory
preflight stopped before any worker started. Sharing another 90 identical large
payload/evidence JSON copies across terminal campaigns verified 302 hashes and
freed **213.84 MB**. These operations preserve every path and byte; the staged
screen was excluded and its unchanged setup resumed afterward. Receipts:
`artifacts/parakeet-ordered-wide-blocks-runtime-links-20260923` and
`artifacts/parakeet-closed-evidence-links-20260923`.

The uniform-mask experiments retired another **135.45 MB across 552 closed VM
build files**: generated package caches and duplicate instruction exports from
V1/V2/V3. All 21 cached package-archive copies matched originals retained in
the protected feed. Complete exports remain verified locally, and every immutable
payload path remains intact. V1/V2
retirement freed91.05MBof tmpfs; V3 freed45.49MB. The latter allowed an unchanged
staged numerical run to pass its original memory preflight before any worker
started. Receipts:artifacts/parakeet-scalar-where-build-retention-20260924 and
artifacts/parakeet-scalar-where-v3-build-retention-20260924. No active worker or
model was removed.

Before the uniform-mask performance screen, closed-only deduplication verified
288 files and shared157 identical binary/evidence copies, freeing **68.54 MB of
tmpfs**. Every path and byte remains unchanged; all recorded owners were terminal.
Receipt: `artifacts/parakeet-scalar-where-closed-links-20260924`. This maintenance
finished before the screen launched.

M57 preparation retired another **45.14 MB across 184 closed VM build files**,
keeping verified package originals and the complete instruction export locally.
Before build and screen staging, two closed-only deduplication passes verified
363 and 487 paths and shared 65 and 92 identical files, freeing **32.01 MB** and
**41.91 MB** of tmpfs. No maintenance overlapped a measured worker. Receipts:
`artifacts/parakeet-provider-where-build-retention-20260924`,
`artifacts/parakeet-provider-where-prebuild-links-20260924` and
`artifacts/parakeet-provider-where-prescreen-links-20260924`.

M58/M59 preparation shared another 32 identical closed runtime files after
verifying 524 paths, then 557 identical evidence metadata files after verifying
1,236 paths. This freed **15.37 MB** and **83.65 MB** of tmpfs with every path
and byte retained. Receipts: `artifacts/parakeet-provider-where-postscreen-links-20260924`
and `artifacts/parakeet-dense-where-evidence-links-20260924`. The latter recovered
a pre-worker memory refusal without lowering the bound or repeating a worker.

After the dense mixed-mask build closed, cleanup removed **45.17 MB across
184 temporary VM files**, freeing **45.54 MB** of tmpfs. Seven package archives
match the retained feed originals, and the full instruction export is verified
locally. All immutable VM inputs and current runtimes remain. Receipt:
`artifacts/parakeet-dense-scalar-where-build-retention-20260924`.
This cleanup completed before the numerical qualification run started.

Before the extended identical-binary Where control, sharing **10,743 identical
closed source, build and evidence files** verified 21,353 paths and freed
**250.25 MB of tmpfs**. Models, fixtures and capture directories were excluded.
All affected campaign owners were terminal; every path and byte remains exact.
Receipt: `artifacts/parakeet-dense-where-stability-headroom-20260924`.
The operation finished before the control was staged or launched.

Before the exact-consumer Where diagnostic, idle VM maintenance retired
**249.48 MB across ten duplicate files**: two complete compiler exports, four
clock journals and four result JSON files from the rejected control. Every
file has a hash-verified complete local copy; the full collection archives and
all 686,400 control clocks remain retained. All owners were terminal and every
immutable payload dependency was protected. Receipt:
`artifacts/parakeet-where-control-remote-retention-20260924`.
The adapter configuration records the exact ten-file scope; the shared
retirement implementation has an older generic five-export description.

After collection, four temporary local inspection copies totaling **4.13 MB**
were removed after matching them to the complete retained stdout/native bodies.
No canonical evidence was removed. Receipt:
`artifacts/parakeet-where-control-live-inspection-retention-20260924.json`.

Before M61 native qualification, idle VM retirement removed **195.44 MB** in
eight duplicate M60 clock/result files. Every complete local result, journal,
archive and native body remains hash-verified; immutable VM inputs are protected.
Receipt: `artifacts/parakeet-where-diagnostic-remote-retention-20260924`.

Before the balanced-warmup control, another **97.74 MB** in four closed VM
clock/result copies was retired after verifying complete local originals and
all immutable dependencies. Receipt:
`artifacts/parakeet-where-balanced-remote-retention-20260924`.
This maintenance finished before the control was staged or launched.

During the M66 graph comparison, the supervisor waited between GPT-2 processes
just below its unchanged 11 GiB memory preflight. With 71 workers complete and
no live worker, maintenance retired one **20.31 MB** VM transfer archive from
the closed selected-profile campaign. Its complete local `payload.tar.gz`
remains verified against closure `e6a94afd`; all 76,719 paths referenced by VM
payload, stage and observer manifests were protected. The final graph worker
then started normally. No timing process, limit, input or score was changed.
Receipt: `artifacts/parakeet-profile-transfer-retention-20260924/closed.json`.

During the focused e5 runtime diagnostic, maintenance reclaimed **175.92 MB**
in seven compiler exports from terminal VM campaigns. Every complete local
export and collection remains hash-verified; all 76,813 immutable input paths
were protected. Receipt: `artifacts/e5-diagnostic-compiler-retention-20260924`.
The preceding attempt to retire old GPT-2 event text found it already absent
and made no mutation (`artifacts/e5-startup-retirement-refusal-20260924.json`).

After all four e5 inference captures finished, decoded event text is collected
incrementally. Each completed export is compressed during transfer, verified
against its original raw SHA-256 and size, then its VM duplicate is removed.
Raw traces and summaries remain on the VM. Final local collection restores all
four original event texts and records `incremental-event-collection.json`;
no event, call or original limit is omitted. Transfer receipts and archives
are in `artifacts/e5-runtime-event-transfers-20260924`.

Between numerical workers of the focused e5 successor, retirement reclaimed
**130.49 MB** from four closed runtime-diagnostic trace copies on the VM.
Every raw trace, full event text and collection archive remains verified locally.
All 76,853 immutable VM paths were protected, and no inference worker was live
during deletion. Receipt: `artifacts/e5-diagnostic-trace-retention-20260924`.

After composition root qualification and the layout observer build were terminal,
retire the root's generated package cache and duplicate instruction export:
626 files, **122.84 MB**, retaining all 24 package archives and the exact local
instruction export. Retire the closed composition V2 build's generated package
cache as well: 584 files, **85.66 MB**, retaining all 20 package archives.
The second operation checked 77,581 protected immutable paths across payload,
stage and specification manifests. All source, runtime inputs and local build
collections remain unchanged. These operations provide headroom for the original
11 GiB capture preflight; no model call was running or retried. Receipts:
`artifacts/parakeet-composition-root-retention-20260924` and
`artifacts/parakeet-composition-build-cache-retention-20260924`.
