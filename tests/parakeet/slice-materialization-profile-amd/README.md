# Matched complete Slice/Reshape profile

Run only after the slice candidate passes complete native/public model
qualification. Compare selected Core `672e5f30` and candidate `bafdb006` on the
same twenty original clips with the already reviewed diagnostic Data observer
`a2a0b490`. Core products and the Data observer are reused unchanged. Rebuild
only the existing observed consumer to accept an explicitly checked Core hash;
all other consumer methods, flags and public surface must remain exact.

Two fresh CPU2 processes execute selected wall profile, then candidate wall
profile. Each preserves one warmup and three measured corpus passes and all
original request checks. Reconcile every graph call and node clock; compare all
24 complete Slice_1/Reshape_7 pairs, including both nodes in each engine. Record
all clips and all other nodes, not a favorable subset. All 160 requests must
match references and each other exactly.

The prospective causal test is **at least 80% lower aggregate time in the 24
complete pairs**. The earlier matched selected/ORT profile measured 2.962s versus
0.024s; bulk region copying should remove most of the coordinate-translation
cost. If it does not, reject this mechanism as tested. This diagnostic threshold
does not admit a release speedup: only the original unprofiled application
comparison can establish the required >=3% corpus gain and <=5% per-clip
regression. The unchanged observer previously added 0.68% complete wall time;
these new profiled clocks retain their own observation overhead, which is not
subtracted. This stage does not independently measure candidate profiler overhead.

Build limits: 2 GiB available/1 GiB tmpfs before each command, 3 GiB owned RSS,
180 seconds/command. Capture: 11 GiB available/2 GiB tmpfs, 12 GiB owned RSS,
900 seconds/process. Keep 1 GiB remaining memory/tmpfs and stage output below
512 MiB. CPU0 monitors exact PID/birth owners; require foreign CPU fraction <=1%.
No concurrent VM work, new model copies or runtime changes.

Use `C:/Python313/python.exe -X utf8 -B` from the repository root with `run.py
prepare`, `stage`, `launch build`, `observe build`, `collect build`, then
`review_build.py`; only its admission allows `launch capture`. Observe until
terminal, collect, then run `audit.py`. All namespaces are exclusive-create.

Artifacts: `artifacts/parakeet-slice-materialization-profile-amd-20260924` and
`/dev/shm/lokad-parakeet-slice-materialization-profile-20260924`.
