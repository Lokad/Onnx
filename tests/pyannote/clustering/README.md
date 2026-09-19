# Managed Community-1 speaker clustering

`Community1Clusterer` in `Lokad.Onnx.Data` connects WeSpeaker embeddings to consistent speaker groups using the automatic pyannote Community-1 configuration. It implements the learned PLDA transform, centroid-linkage initialization, VBx refinement, weighted centroids and constrained assignment. It does not yet extract segmentation windows or produce diarization timelines.

```csharp
var clusterer = new Community1Clusterer("models/pyannote-community-1/plda.json");
Community1ClusteringResult groups = clusterer.Cluster(
    embeddings, binaryActivity, cancellationToken);
// embeddings: one WeSpeakerEmbedding per chunk/local-speaker pair, row-major.
// binaryActivity: Tensor<bool>[chunks, frames, localSpeakers], 1..3 local speakers.
// groups.Labels: row-major [chunks, localSpeakers]; -2 means unassigned.
// groups.Centroids: owned read-only 256-value double vectors, indexed by labels.
```

Only valid embeddings with at least 20 percent clean single-speaker frames train the clusters. `InsufficientFrames` entries have no vector and are excluded. Completed entries require 256 finite values and a nonzero norm; an undefined/nonfinite derived norm is rejected. No usable training data returns empty centroids and `-2` labels. One training vector defines one centroid and labels active valid entries with it. Inactive, invalid and unmatched entries remain `-2`. Completed clustering is not a speech-accuracy or speaker-identity guarantee.

The standard automatic path uses threshold `.6`, Fa `.07`, Fb `.8`, at most 20 VBx iterations, objective-improvement stopping at `1e-4`, and retained priors above `1e-7`. Centroid cuts account for decreasing merge heights. Output labels follow their first assigned occurrence; unused centroids follow afterward. Forced speaker counts and their KMeans fallback are outside this component's automatic-count contract.

Each request owns scratch and results, preserves its inputs, and checks cancellation during clustering. Instances may serve independent concurrent requests. Limits are 12,288 input embeddings, 4,096 selected training embeddings and 8,388,608 activity values. Hierarchy storage grows quadratically; these limits are resource bounds, not evidence that every maximum-size workload meets a latency target.

## Prepare local learned parameters

The selected pipeline is [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1/tree/3533c8cf8e369892e6b79ff1bf80f7b0286a54ee), with pyannote.audio 4.0.0 at `a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a`. Obtain its pinned `plda/plda.npz` and `plda/xvec_transform.npz` locally, retaining the model's attribution and license. Preparation verifies both full SHA-256 hashes, shapes, dtypes and finite values, then precomputes the fixed whitening decomposition:

```powershell
python -m pip install -r tests/pyannote/clustering/requirements.txt
python eng/prepare-community1-plda.py --model-directory models/pyannote-community-1/plda --output models/pyannote-community-1/plda.json
```

The preparation tool requires NumPy 2.2.4 and SciPy 1.16.3, checks the decomposition residual, performs no model downloads and refuses existing output paths. The JSON includes parameters and provenance. Runtime uses managed math and that local JSON only; it has no NumPy, SciPy, Torch or native ORT dependency. Source attribution and licenses accompany Data.

## Independent qualification

Ordinary tests use tiny independent mathematical examples and generated identity parameters. They cover centroid-distance inversions, exact cut boundaries, duplicate points, exhaustive assignment optima, a closed-form VBx posterior, clean-frame thresholds, no-data behavior, malformed inputs, resource limits, cancellation, concurrency and ownership.

The optional lane executes the actual pinned upstream functions and compares every transform/refinement/centroid/assignment value. It contains 15 complete clustering cases, including native and managed recorded vectors, and 15 hierarchy cases. The managed replay calls the compiled product primitives and public API; it does not compile a private copy of product math. Initial cluster labels are compared under one recorded bijection, and every later output uses that same mapping. Public labels are canonicalized by first appearance. All 188 arrays / 229,714 values pass locally, maximum `2.98024e-8`, including 30 public API requests and cancellation/recovery. This does not resolve the separate frontend/segmentation numerical failures or qualify complete audio diarization.

```powershell
python tests/pyannote/clustering/generate_reference.py --reference-source external/pyannote-audio --model-directory models/pyannote-community-1/plda --speaker-reference artifacts/speaker-reference --managed-speaker-result artifacts/speaker-result.json --output artifacts/clustering-reference-new
dotnet build tests/pyannote/clustering/ClusteringReplay.csproj -c Release --tl:off --nologo -v minimal
dotnet tests/pyannote/clustering/bin/Release/net10.0/ClusteringReplay.dll artifacts/clustering-reference-new artifacts/clustering-result-new.json
python tests/pyannote/clustering/audit.py --reference artifacts/clustering-reference-new --result artifacts/clustering-result-new.json --output artifacts/clustering-audit-new.json
```

Use the [speaker reference recipe](../speaker/README.md) for the preceding artifacts. The source checkout must contain the pinned pyannote tag. Every destination must be fresh. The independent auditor checks raw arrays, training selection, partitions, complete coverage, canonical labels, repeated centroids and numerical summaries. Reference generation and Python packages are optional development tools; ordinary tests and product inference do not download assets.
