# Local artifact retention

The 2026-09-23 cleanup removed **119.94 GB across 210,772 obsolete artifact
files**. The repository then contained **37.15 GB**, including 23.74 GB in
`artifacts` and 10.79 GB in canonical `models`. Counting the worktree's junction
to those same models again gives **47.94 GB**. Both measurements are below the
requested 50 GB limit; units here are decimal GB.

After retaining the next Parakeet candidate's build, numerical, component,
application, cross-model and follow-up diagnostic evidence, the inventory
measured **38.62 GB**. After collecting and publishing the complete warmed
comparison and the closed padding build/screen, the latest inventory is
**39.27 GB**, including **25.84 GB** in artifacts. Counting the model junction
again produces **50.05 GB**, but that includes the same 10.79 GB of canonical
models twice; it is not another stored copy. The actual repository remains
below 50 GB. Its receipt is
`artifacts/repository-retention-20260923/m49-closed-size.json`. The padding
diagnostic reused existing binaries and copied no models. Recheck after
subsequent collections; the actual repository has about 10.73 GB of headroom.

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
