# Full Parakeet correctness for guarded slice materialization

Qualify reviewed Core `bafdb006` with unchanged Data `065b7a7f` against selected
Core `672e5f30`. Actual layouts passed `664cde30`, compiled scope `aa34820a`, and
369 tensor tests pass in each verified instruction mode at `2ca35d78`. The two
test-setup failures remain retained; no candidate performance is admitted yet.

Reuse the original TranscribeReplay `335ca09d` and AudioBenchmark `7eca033a`
consumers, canonical models, twenty PCM clips and complete native references.
Nothing is rebuilt. Eight CPU2 processes execute selected native/public and
candidate native/public in ordinary mode, then the same four jobs with
`DOTNET_EnableAVX512=0`. This exact group setting was verified by the preceding
hardware assertion; the consumers themselves record settings, not a new hardware
probe. CPU0 monitors every process and thread.

Each native worker must pass all 784 arrays / 3,090,494 values at native scaled
error <= 1e-4, complete decisions, rejected requests, cancellation and recovery.
Candidate arrays must match selected bytes within each mode. Each public worker
checks all twenty complete results, immutable PCM and held-output ownership.
The native/public auditors and checks are byte-identical to the previous model
qualification; numerical assertions, consumers and workload are unchanged.

Prospective resource decision, made before staging: require **11 GiB available
memory**, rather than the previous lane's 12 GiB. The idle VM currently has
12,423,487,488 bytes available, below 12 GiB. The identical prior eight-job
qualification peaked at 8,505,827,328 bytes owned RSS; the recent original
application captures also completed under an 11 GiB preflight. This independent
stage retains the **12 GiB RSS cap, 1 GiB remaining memory guard**, 3 GiB tmpfs
preflight, 1 GiB remaining tmpfs, 1 GiB output/job, 2 GiB total artifact cap,
1,800 seconds/job and four-hour campaign cap. Bounds are frozen before any run;
no failed launch or numerical criterion is waived.

New files may hardlink immutable closed inputs. Product entries are unlinked
before replacement, never overwritten in place. Collection exports hardlinks as
regular members and independently verifies all hashes. All raw arrays/results,
resource samples and failures must be retained. No timing gate may follow failed
model correctness.

Run `C:/Python313/python.exe -X utf8 -B` from the repository root with
`run.py prepare`, `stage`, `launch`, `observe`, `collect`, then `audit.py`.
Observe only until terminal; never reopen closed stages. Namespaces:
`artifacts/parakeet-slice-materialization-models-amd-20260924` and
`/dev/shm/lokad-parakeet-slice-materialization-models-20260924`.
