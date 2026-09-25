# Full Parakeet correctness after dispatch isolation

Preparation requires the admitted combined graph qualification for Core e07a4518.
It independently recomputes the seven original passing cases and the diagnosed
short-e5 correction from their complete clocks. The original failed campaign
remains unchanged. Both source campaigns must be terminal, the corrected case
and all combined gates must pass, and exact product identities must agree.
No model staging or inference starts while that qualification is missing or failed.

Compare the successful direct-depthwise parent Core40260aef with Coree07a4518;
both use Data01e9e784. Reuse the original eight-process full-model protocol,
consumers, checks and auditor. Only product provenance, transport, staging and
the auditor's descriptive product labels change.

Each product and instruction mode checks 784 arrays / 3,090,494 values and 20
complete public transcriptions. Require bit-identical parent arrays and complete
public results, native scaled error <=1e-4, immutable inputs and independent held
outputs. Ordinary and AVX512-disabled modes total 3,136 arrays, 12,361,976 values
and 80 public requests. This stage produces no performance score.

The original limits remain: CPU2 computes and CPU0 monitors; 11 GiB available RAM
and 3 GiB tmpfs before each worker; 12 GiB RSS; 1 GiB remaining RAM/tmpfs; 1 GiB
output per job; 2 GiB stage; 1,800 seconds per worker and four hours total.
Reuse all models, reference tensors, consumers and dependencies already present.

From the repository root use `C:/Python313/python.exe -X utf8 -B` with this
directory's `run.py prepare`, `stage`, `launch`, `observe`, `collect`, then
`audit.py`. Observe the same owner until terminal, collect/audit once and preserve
any failure. Full application performance, Pyannote and root/package qualification
remain required before source or BENCHMARK.md promotion.
