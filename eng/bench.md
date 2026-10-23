# Routine CPU comparisons

`bench.py` runs the current validated workers against two prepared cores, in
`control,candidate / candidate,control` order. It retains 33 calls from every
engine process, numerical checks, input/model/binary identities and process
accounting. It requires no A/A calibration and adds no cooldown. Each worker
keeps its existing warmup: 30 seconds of conditioning for e5 and at least
10 seconds plus 32 calls for Gemma. Engines run in separate processes on one CPU.

Prepare committed cores locally, then run on the machine with the local models:

```powershell
pwsh -NoProfile -File eng/build-common-campaign.ps1 -L0Ref <control-commit> -L1Ref <candidate-commit> -Output artifacts/e5-prepared
python eng/bench.py --model e5 --prepared artifacts/e5-prepared --output artifacts/e5-comparison --cpu 2

pwsh -NoProfile -File eng/build-embeddinggemma2-benchmark.ps1 -L0Ref <control-commit> -L1Ref <candidate-commit> -Output artifacts/gemma-prepared
python eng/bench.py --model gemma --prepared artifacts/gemma-prepared --fixtures <fixtures-directory>/fixtures.json --output artifacts/gemma-comparison --cpu 2
```

The defaults are `e5-30tok` and `gemma2-128tok`, with 768-dimensional Gemma
vectors. Add `--case gemma2-128tok --case gemma2-512tok` for both widths, or
`--pairs 4` for more repetitions. `--root` selects the checkout containing local
model assets; its `eng` directory need not contain this command. Gemma fixtures
come from the existing [reference lane](../tests/embeddinggemma2/README.md).
Uncommitted source changes are absent from prepared Git-archive builds.

`report.json` contains every timed call, per-process medians, managed/ORT ratios,
paired candidate/control ratios and process variation. The immutable worker
files retain full numerical evidence. Completed workers survive a later failure.
Output directories must be new. An exit failure means missing/failed worker
validation; performance differences and ordinary variation are reported rather
than turned into a calibration refusal.

Routine and strict commands share `benchmark_worker.py` for staging/processes,
`e5_worker.py` for e5 launch/validation, and `gemma_worker.py` for Gemma fixtures
and workers. The routine command does not import strict campaign frontends or
their scorer. Strict evidence inventories include the extracted worker sources;
historical inventories remain readable with their original archived producers.

These are engineering observations. Small changes within the reported variation
are inconclusive. Keep the existing strict
[isolated protocol](isolated-evidence.md) and [Gemma protocol](../tests/embeddinggemma2/README.md)
for historical reproduction and release performance claims; short comparisons
do not update the published benchmark table automatically.

Importer assembly compatibility: the 0.4.0 Protobuf extraction moved parsing into
`Lokad.Onnx.Protobuf`. For newly prepared comparisons, choose both source refs
from the same importer era (`-L0Ref` and `-L1Ref`). Preparation builds a separate
adapter from the baseline source when present and stages it unchanged with both
cores. Integrated/separate importer mixes are rejected because one compiled
caller cannot span that assembly move. Existing prepared historical runners and
results remain valid with their original cores.
