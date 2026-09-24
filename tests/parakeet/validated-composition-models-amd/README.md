# Full Parakeet correctness for the exact admitted-source composition

Run only after both individual application comparisons pass and the composed
Core passes compiled composition review and all369tensor cases in both verified
instruction modes. Reuse measured M64 Data cc37b19e and the newly built Core;
selected remains Core672e5f30/Data065b7a7f. Their exact identities are frozen in
the stage. No additional source, arithmetic, tuning or consumer change.

The native/public consumers and auditors, numerical checks, worker and resource
protocol are byte-identical to slice-materialization-models-amd. Only candidate
products, prerequisite composition and manifest labels change. Keep all784native
arrays/3,090,494values and20publicclips for each role/mode. Candidate must match
selected bytes and complete public results exactly; both retain the original
native1e-4 scaled bound and input/held-output checks. These fresh complete
trajectories test interactions between the separately proved implementations.

Eight fresh processes run selected-native/public and candidate-native/public
in ordinary mode, then the same four with DOTNET_EnableAVX512=0. No scored
performance is inferred. The original six-process application comparison and
all shared-model/root/release gates remain afterward.

Use `C:/Python313/python.exe -X utf8 -B` with `run.py prepare`, `stage`, `launch`,
`observe`, `collect`, then `audit.py` from this directory. Local artifacts are
`artifacts/parakeet-validated-composition-models-amd-20260924`; remote namespace
is `/dev/shm/lokad-parakeet-validated-composition-models-20260924`. Freeze before
staging, refuse namespace reuse, collect only terminal PID/birth owners.

Unchanged bounds:11GiBavailable/3GiBtmpfs preflight,12GiBRSS, at least1GiBremaining
memory/tmpfs,1GiBoutput/job,2GiBstageartifacts,1800s/job,fourhours/stage. CPU2
computes, CPU0 monitors. Reuse canonical models through immutable hardlinks;
unlink product targets before replacing, never modify a source link in place.
