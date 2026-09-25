# Graph qualification of the packed-dispatch relocation

Compare qualified release `f95a13c5`, the single relocation `e07a4518`, and fresh
Microsoft ORT. The candidate restores release's shared batched dispatcher and
passes all 89 existing focused contracts. Its Parakeet parent remains the
successful 53.48-second direct-depthwise candidate; no new application score is
claimed here.

Reuse the entire original graph comparison: all eight cases, 72 processes,
41,112 calls and 8,640 measurements. Preserve every numerical bound, ownership
check, warmup count, clock, repeatability control and 5% regression threshold.
The worker, protocol, consumers, statistical code and full auditor are reused
unchanged from `../packed-final-row-graphs-amd`. That parent's short-e5 failure
remains recorded; this candidate is an actual three-method change.

Only provenance, transport and staging change. Existing inputs and consumers
are hardlinked, with verified identities; candidate Core is replaced in both
ordinary and e5 runtimes, and both manifests declare its actual hash. The
original memory, CPU affinity, per-job and whole-campaign limits remain intact.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root with this
directory's `run.py prepare`, `stage`, `launch`, `observe`, `collect`, and
`audit.py`. Observe the same owner until terminal. Preserve failures; do not
repeat the campaign unchanged. Full Parakeet, Pyannote and release integration
remain separate requirements after this graph comparison.
