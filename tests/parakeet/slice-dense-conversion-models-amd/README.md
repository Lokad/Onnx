# Full Parakeet correctness for the single slice-copy override

Compiled review349ae125 proves all 3,253 original Core methods unchanged and
only the inherited dense-conversion override added. Tensor closurea4d86d05
passes 395 cases in each actual instruction mode. Compare current Coref95a13c5
against candidate49c3a958, both using the same ordinary Dataa893952f.

Reuse the complete M70 protocol, consumers, worker and numerical/public auditors
unchanged. Only prerequisites, product bindings, namespace and provenance labels
change. Each product runs 784 arrays / 3,090,494 values against the pinned ORT
reference and twenty full public clips, in ordinary and disabled AVX512 modes.
Require exact current arrays and public results, native scaled error <=1e-4,
unchanged inputs and independently owned held outputs. All 3,136 arrays,
12,361,976 values and 80 public requests remain. These are correctness checks;
fresh ORT application timing follows separately after matched attribution.

Eight serial CPU2 workers, monitored on CPU0, retain the original limits:
11 GiB available / 3 GiB tmpfs before each job, 12 GiB RSS, 1 GiB remaining
memory/tmpfs, 1 GiB output/job, 2 GiB stage output, 1,800 seconds/job and four
hours total. No compilation or model download. Reuse canonical model assets
and consumers through verified hardlinks, unlinking replacements first.

Use Python3.13 -X utf8 -B with run.py prepare, stage, launch, observe and collect;
then audit.py. Tools freeze at preparation. Collect only terminal PID/birth
owners, preserve every failure, and never repeat a completed worker. Store audit
stdout outside its campaign directory. Local artifacts use
artifacts/parakeet-slice-dense-conversion-models-amd-20260925; the VM uses
/dev/shm/lokad-parakeet-slice-dense-conversion-models-20260925.
