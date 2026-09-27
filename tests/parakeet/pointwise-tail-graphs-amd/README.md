# Graph regression checks for the admitted pointwise remainder candidate

Compare current Core `47984318`, unchanged candidate `7cac6788` and the retained
ORT 1.29.0 binary. Require complete Parakeet correctness, admitted Parakeet
application performance, shared/e5 correctness and Pyannote correctness before
preparation. Pyannote closure `f284cc72` passes and is bound before freezing tools.
Preserve the rejected component screen and its runtime diagnosis unchanged.

Reuse ReleaseBenchmark `d827e3b9` for ordinary cases, `0b228b2d` for 30-token e5
and `e437850d` for eight-token e5, with their original method/flag proofs. Product
compatibility reconciles all 3,983 original Core/Data methods: one packed-kernel
body changes and two internal remainder helpers are added, with all original
flags/public bindings and all Data methods preserved. These consumers load Core
only. No consumer or product build occurs.

The original 72 jobs cover five e5 inputs, DINOv3, ResNet50 and GPT-2. Keep three
verification workers and six timing processes per case in current, candidate,
ORT, ORT, candidate, current order. Retain 6,000 warmups for eight-token e5,
1,200 for 30-token e5 and 600 otherwise, then 180 measurements. Preserve every
one of the 73,512 calls, 8,640 measured clocks and 72 setup intervals.

Workers, numerical checks, scoring and complete auditor are unchanged. Keep
finite outputs, native scaled error <=1e-4, exact candidate/current arrays,
input/output ownership, <=5% regression and <=10% process disagreement. The
four-export/eight-case census proves coverage, without a runtime-dispatch claim.
No clock filtering, new flags, model downloads or implementation variants occur.

Link retained references and three existing runtimes from graph closure `f7b0a361`.
Resolve retained paths before hardlinking, using a digest-verified copy when the
filesystems differ. Preserve exact payload/collection metadata, terminal owner
checks and every linked/external input. Preserve CPU2 compute / CPU0 monitoring,
11 GiB RAM / 3 GiB tmpfs preflight, 8 GiB owned RSS, 1 GiB remaining RAM/tmpfs,
900 seconds/job, 2 GiB campaign files and four hours overall. Run only one VM
workload; inspect headroom before staging the complete output set.

After successful prerequisite binding and source freeze, prefix commands with
`C:/Python313/python.exe -X utf8 -B` from repository root:

    -m unittest discover -s tests/parakeet/pointwise-tail-graphs-amd -v
    tests/parakeet/pointwise-tail-graphs-amd/run.py prepare
    tests/parakeet/pointwise-tail-graphs-amd/run.py stage
    tests/parakeet/pointwise-tail-graphs-amd/run.py launch
    tests/parakeet/pointwise-tail-graphs-amd/run.py observe

Follow the same owner to terminal, then collect and audit once. Keep console
output outside the closed artifact and never replay completed phases.
Local: `artifacts/parakeet-pointwise-tail-graphs-amd-20260927`.
VM: `/dev/shm/lokad-pwt-graphs-20260927`.
Complete Pyannote application regression, portable boundary tests and actual-root
tests/package qualification remain required before release promotion.
