# Finish only the unstarted M76 counter jobs

The first selected-512 counter worker completed all20 clips. Before candidate-512
started, the immediate11GiB preflight guard saw11,618,447,360 available bytes and
refused launch. The original campaign is terminal with code1; candidate-512 has
no PID, samples or outputs. Neither AVX512-disabled worker started. Preserve
that complete failure and the successful selected-512 result.

This recovery binds all18 reviewed runtime files without compilation. Retain
the selected-512 outputs byte-for-byte and run only candidate-512, selected-256
and candidate-256. Before each launch, retain observations while waiting up to
900seconds for the existing11GiB guard, as the full-model protocol already does.
No numerical, scratch/copy, resource or acceptance threshold changes. The counter
consumer itself is unchanged. Count1740 avoided packs and609 reconstructions per
20-clip corpus, with exact output hashes and all per-request counter differences.

The audit independently checks the retained successful worker, the original
unstarted failure and the three new workers. All four role/mode combinations
must pass. The original selected-512 clock is never rerun or replaced.

Use run.py prepare, stage, launch build, observe build, collect build, then
review.py build. This build phase only binds binaries. Then launch capture,
observe capture, collect capture, and review.py capture. Store audit stdout
outside the artifact namespace; tools freeze at preparation. Local artifacts
use parakeet-owned-packed-weight-counters-resume-amd-20260925, and VM files use
/dev/shm/lokad-parakeet-owned-packed-weight-counters-resume-20260925.
