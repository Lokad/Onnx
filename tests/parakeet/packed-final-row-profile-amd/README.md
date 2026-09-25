# Attribute the remaining Parakeet cost on the actual M78 product

Prepare the reviewed phase/wall observer from exact M78 Data source. The old
diagnostic Data assembly predates encoder.PrepareOwnedMatMulWeights(). Reusing
it would disable the optimization being measured, despite a matching Core.

prepare_source.py preserves all 22 M78 Data source files, inserting only the
existing observation scope at the private Execute helper's entry. Removing the
scope must recover the original source bytes. The constructor's preparation call
remains intact. The observer implementation and project/bridge sources match the
closed M65 observer review. Reuse SampledAudio 38ab5c7e unchanged: its explicit
expected Core/Data environment checks already support these exact products.
The runtime base contains measured M78 Core 49901366 and original Data 01e9e784.

This step is local source preparation only. It does not build, infer, promote
root source or assign timing. Run Python 3.13 -X utf8 -B -m unittest discover -s
tests/parakeet/packed-final-row-profile-amd, then prepare_source.py once. It
refuses an existing artifacts/parakeet-packed-final-row-profile-source-20260925.
After preparation, these source-preparation files and outputs are frozen.

After graph, direct Parakeet release, Pyannote and root/package admission, build
only diagnostic Data and the inventory bridge on the idle AMD VM. Keep Core and
the reviewed request runner byte-identical. Before inference require all 164
runner methods unchanged and all 697 original Data methods preserved except the
one Execute hook, including constructor, public declarations and method flags.
Reverse the compiled scope to prove the original Execute instructions, branches
and exception behavior. Keep the original observer's 50 added internal methods.

Then run three fresh processes: original M78 Data control, observed phase-only,
observed wall-profile. Each runs all 20 clips, one warmup and three measured
passes: 240 complete requests. Retain every public/native result, input/held-output
check, graph/node interval, process identity and resource observation. Quantify
both observer overhead steps; subtract neither. Keep CPU 2 work/CPU 0 monitoring,
11 GiB available/2 GiB tmpfs inference preflight, 12 GiB RSS, 1 GiB remaining,
900 seconds per process and 512 MiB output. Build bounds remain 2 GiB available,
1 GiB tmpfs, 3 GiB RSS and 180 seconds per command. One VM workload at a time.

Reconcile every phase and node. Match all 217 ORT constant projections through
actual graph edges and shapes, including the 48 fused half-scale operations:
265 managed nodes in the established mapping. Preserve all dynamic products and
other work separately; count shared preparation once. The historical native
profile and its exact installed binary remain explicitly dated reference evidence.
An updated managed profile is attribution, not a fresh cross-engine score.

Select one next optimization only if complete measured groups explain the
remaining gap and source/generated-code evidence identifies an applicable cause.
Row count, vector width or a profiler sample share alone predicts no speedup.
The compiled-review, VM capture and comparison adapters still need implementation;
the source preparation does not qualify an executable observer.
