# Full Parakeet correctness for owned packed weights

The corrected candidate passes compiled review5a84c53b, focused contracts27/27/2
at0f8bdc94 and actual-model census577a07a6. Exactly87 intended weights are replaced;
all37 existing maps and the longest public transcript remain intact in both modes.
Compare selected isolatedM73 Core49c3a958/Dataa893952f against candidateM76
Core82c02785/Data3f80f8cb. Neither isolated candidate is admitted for release;
the prior e5 repeatability failures remain unresolved.

Reuse the complete M70 protocol, consumers, worker and numerical/public auditors
unchanged. Only prerequisites, product bindings, namespace and provenance labels
change. Each product runs784 arrays /3,090,494 values against pinned ORT truth
and20 complete public clips, in ordinary and disabled AVX512 modes. Require exact
selected arrays and public results, native scaled error<=1e-4, unchanged inputs
and independently owned held outputs. All3,136 arrays,12,361,976 values and80
public requests remain. These are correctness checks, not performance scores.

The unchanged TranscribeReplay consumer constructs ParakeetTranscriber before
accessing its private encoder, so the candidate Data constructor is exercised.
The full public consumer exercises ordinary construction and Transcribe calls.
Do not replace either consumer with a plain graph loader that bypasses preparation.

Eight serial CPU2 workers, monitored on CPU0, retain the original limits:
11GiB available /3GiB tmpfs before each job,12GiB RSS,1GiB remaining memory/tmpfs,
1GiB output/job,2GiB stage output,1,800seconds/job and four hours total.
No compilation or downloads. Model assets and consumers reuse verified hardlinks.

Use Python3.13 -X utf8 -B with run.py prepare, stage, launch, observe and collect;
then audit.py. Tools freeze at preparation. Collect only terminal PID/birth
owners and preserve every failure. Never repeat a completed worker. Store audit
stdout outside its campaign. Local artifacts use
artifacts/parakeet-owned-packed-weight-models-amd-20260925; VM files use
/dev/shm/lokad-parakeet-owned-packed-weight-models-20260925.
