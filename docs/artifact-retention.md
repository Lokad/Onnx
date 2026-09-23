# Local artifact retention

The 2026-09-23 cleanup removed **119.94 GB across 210,772 obsolete artifact
files**. The repository then contained **37.15 GB**, including 23.74 GB in
`artifacts` and 10.79 GB in canonical `models`. Counting the worktree's junction
to those same models again gives **47.94 GB**. Both measurements are below the
requested 50 GB limit; units here are decimal GB.

After retaining the next Parakeet candidate's build, numerical, component and
complete-model evidence, a fresh inventory measured **37.85 GB**, or
**48.63 GB** counting that junction twice. Its receipt is
`artifacts/repository-retention-20260923/m43-during-application-size.json`.

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
