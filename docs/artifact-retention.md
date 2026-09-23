# Local artifact retention

The 2026-09-23 cleanup removed **119.94 GB across 210,772 obsolete artifact
files**. The repository then contained **37.15 GB**, including 23.74 GB in
`artifacts` and 10.79 GB in canonical `models`. Counting the worktree's junction
to those same models again gives **47.94 GB**. Both measurements are below the
requested 50 GB limit; units here are decimal GB.

The latest inventory is **41.78 GB**, including **28.33 GB** in artifacts,
after retaining M54's build, numerical, component and complete Parakeet
correctness evidence, closing its application and cross-model correctness
comparisons, closing graph and Pyannote application checks, qualifying the actual
root and package, and closing the next matrix-blocking candidate's build,
numerical qualification and rejected performance screen. A fresh inventory
confirms **41,776,159,353 bytes**, with **8.22 GB**
of headroom. Receipt:
`artifacts/repository-retention-20260923/m55-closed-size.json`.
Counting the model junction again gives 52.56 GB, which counts the same 10.79 GB
of canonical models twice. The inventory excludes that directory alias and
counts each actual file path, including hidden Git files and retained worktrees.
No new model copies were needed. Recheck after subsequent collections.

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
