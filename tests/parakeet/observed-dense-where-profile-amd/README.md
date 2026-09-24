# Test the observed masking mechanism in complete Parakeet profiles

Run after all current-composition native/public model checks pass. Reuse the
exact reviewed wall-observer Data `a2a0b490` and consumer `38ab5c7e`. Their
original Data methods and flags match all 697 current-release methods, and the
original Data source files remain exact. The existing consumer already accepts
the explicitly supplied Core hash. No new observer or consumer build occurs.

Two fresh CPU 2 processes profile current Core `37c24375` and candidate
`f95a13c5` on the same twenty clips, one warmup and three measured-label passes
each. Both use the same observed Data/consumer and the current release manifest.
All 160 complete requests, public/native checks, graph identities and node
intervals must reconcile. Profile clocks include observer overhead and are not
scored application latency; no overhead is subtracted.

The prospective prediction is **at least 80% lower summed time in all 72 Where
kernels**, with every family's complete input group improving. Reuse the exact
previously matched graph boundaries: the mask's Not output and the Where data
input, through condition expansion/casts and scalar creation to its output.
Count shared ancestors once in the combined total. Report every layer, all
three families, full group totals and all other node clocks. Reconcile the
same original graph and nineteen encoded lengths; do not select favorable nodes.

Capture limits remain 11 GiB available / 2 GiB tmpfs before each process,
12 GiB owned RSS, 900 seconds/process, 1 GiB remaining memory/tmpfs and 512 MiB
stage output. CPU 0 monitors exact owners; require foreign CPU fraction <=1%.
No concurrent VM work or runtime changes. Only passing attribution advances
to the unchanged complete-application comparison and its original gates.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py inspect` for a read-only
observer reuse check. Then run `test_groups.py`, `run.py prepare`, `stage`,
`launch`, `observe`, `collect` and `audit.py`. There is only a capture stage;
no build stage is manufactured. Freeze tools at preparation. Redirect audit
stdout outside the audited directory, and collect only terminal owners.

Local namespace: `artifacts/parakeet-observed-dense-where-profile-amd-20260924`.
VM namespace: `/dev/shm/lokad-parakeet-observed-dense-where-profile-20260924`.
