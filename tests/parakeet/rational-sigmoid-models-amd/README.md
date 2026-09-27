# Full Parakeet correctness for the fixed rational sigmoid

Compare qualified Core 8bb22038 / Data d02dbf55 with already-built rational
Core 946ddfb6 / Data dbe95936. Reuse the exact TranscribeReplay and AudioBenchmark
consumers and all fixtures from the closed padding-model campaign 194eb9a4.
The root inventory proves all 3,282 Core / 697 Data methods and public metadata
equal the product previously qualified with these consumers. The rational
inventory changes only Sigmoid and adds one private helper; Data and public
bindings stay exact. No product, consumer, model or reference is rebuilt.

The operator screen fc8a6d97 remains rejected: 74.992572% weighted saving, 13
repeatability failures and four fallback regression failures. The subsequent
fixed observation c6f0a34e shows equal allocation minima and scalar opcode counts,
with collection-associated tiny-case spikes; double latency is still unresolved.
PLAN permits this correctness-only model check after that diagnosis. Neither the
diagnostic nor this evaluation admits the operator screen or a release.

Reuse the complete eight-process worker, resource protocol and independent
native/public validators unchanged. Both products run normal and AVX512-disabled
modes. Each product/mode checks 784 arrays / 3,090,494 values and 20 complete public
transcriptions: totals 3,136 arrays / 12,361,976 values and 80 transcriptions.
Preserve native scaled error <=1e-4, exact integer outputs, shapes, all decoder
decisions/tokens/transcripts, immutable inputs and independently held outputs.

The sole behavioral audit adaptation replaces the padding-specific cross-product
float byte equality with abs(candidate-current)/max(1,abs(current)) <=1e-4.
Record both hashes, every shape, value count, maximum and worst index; retain
bit equality as an observation. Integers and complete public results remain exact.
The independent ORT comparison keeps its original bound and reference bytes.
Preparation proves the rest of checks.py is unchanged. Tests reject out-of-bound
or nonfinite floats, integer changes, shape/size mismatches and missing outputs,
as well as unrelated compiled-method or implementation-flag changes.

Original bounds: CPU2 computes, CPU0 monitors; 11 GiB available RAM and 3 GiB
tmpfs before each job; 12 GiB owned RSS; >=1 GiB RAM/tmpfs remaining; 1 GiB
output/job, 2 GiB stage, 1,800 seconds/job and four hours overall. The worker's
bounded 900-second preflight wait is unchanged. Do not relax limits or duplicate
model weights. Verify idle owners and reuse canonical fixtures via hardlinks.

The one-time helper eng/retire_sigmoid_model_headroom.py already retired a closed
private cache and byte-identically retained diagnostic output copies, freeing
165,588,992 bytes while preserving 23 canonical offline archives and 577 protected
files. Its closed receipt is artifacts/parakeet-rational-model-headroom-20260927.
Do not run that helper again.

From repository root, prefix commands with C:/Python313/python.exe -X utf8 -B:

    -m unittest discover -s tests/parakeet/rational-sigmoid-models-amd -v
    tests/parakeet/rational-sigmoid-models-amd/run.py prepare
    tests/parakeet/rational-sigmoid-models-amd/run.py stage
    tests/parakeet/rational-sigmoid-models-amd/run.py launch
    tests/parakeet/rational-sigmoid-models-amd/run.py observe
    tests/parakeet/rational-sigmoid-models-amd/run.py collect
    tests/parakeet/rational-sigmoid-models-amd/audit.py

Freeze tools at preparation. Observe only while the original owner is live;
collect/audit once after all owners are terminal. Preserve any failure and do
not rerun completed requests to repair reporting. Audit stdout belongs outside
the artifact directory. Local namespace:
artifacts/parakeet-rational-sigmoid-models-amd-20260927;
VM: /dev/shm/lokad-parakeet-rational-sigmoid-models-20260927.

This produces no performance score. Numerical success would permit a separate
prospective decision about application timing, preserving unresolved fallback risk
and all original application gates. Complete release qualification is still
required before root source or BENCHMARK.md changes.
