# Full Parakeet correctness for the observed-mask composition

Run after compiled scope `7d5296b2` and complete operator numerics `0dfb5dcb`.
Compare current Core `37c24375` / Data `cc37b19e` with candidate Core `f95a13c5`
/ Data `a893952f`. Reuse the qualified model consumers, worker, native/public
auditors, every assertion and resource bound unchanged. Only product bindings,
prerequisites, namespace and manifest provenance labels change.

Each product runs 784 arrays / 3,090,494 values against the pinned ORT reference,
plus all twenty complete public transcription clips, in ordinary and
AVX512-disabled modes. Require exact current-product arrays and complete public
results, the original native scaled-error bound of 1e-4, input immutability and
held-output ownership. Totals: 3,136 arrays / 12,361,976 values and 80 public
requests. These are fresh managed trajectories against retained native truth;
the subsequent scored application comparison also runs ORT afresh.

Eight serial processes use CPU 2, monitored on CPU 0. Unchanged limits are
11 GiB available / 3 GiB tmpfs before each job, 12 GiB RSS, 1 GiB remaining
memory/tmpfs, 1 GiB output/job, 2 GiB stage output, 1,800 seconds/job and four
hours/stage. No compilation, model copies, new runtime flags or timing claims.
Reuse canonical assets through verified hardlinks; unlink replacements first.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`,
`launch`, `observe`, `collect`, then `audit.py`. Freeze tools before preparation;
collect only terminal owners and redirect audit stdout outside the audited
directory. Local namespace: `artifacts/parakeet-observed-dense-where-models-amd-20260924`.
VM namespace: `/dev/shm/lokad-parakeet-observed-dense-where-models-20260924`.
Full-model admission precedes matched profiles and complete application timing.
