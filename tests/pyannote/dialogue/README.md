# Annotated two-speaker dialogue

This optional lane tests the complete Community-1 pipeline on the official
thirty-second pyannote.audio sample and three fixed ten-second excerpts. The
sample contains two annotated anonymous speakers and overlap. The full recording
is the primary labeled observation; the excerpts expose boundary/context effects
and are not three additional independent recordings.

The WAV and RTTM annotations come from `src/pyannote/audio/sample` at upstream
commit `a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a` (pyannote.audio 4.0.0).
[Upstream assets and attribution](https://github.com/pyannote/pyannote-audio/tree/a1ed3bb0440d33d18622e1cb6b138431cfaf4f7a/src/pyannote/audio/sample)
remain authoritative; this repository stores hashes and recipes, not the audio.
WAV SHA256 is `c319b4abca767b124e41432d364fd7df006cb26bb79d09326c487d606a134e6e`;
RTTM SHA256 is `d78fe62c69d8e6dcbb42c26adfce83faccb374c5a1e6d987fe37f85f1c173c87`.

PCM16 samples decode exactly to float32 by division by 32768. Fixed intervals
are `[0,30)`, `[0,10)`, `[10,20)` and `[20,30)` seconds. Cropped annotations are
clipped and shifted into each excerpt's coordinates. The complete clip spans
21 overlapping pipeline windows; each excerpt spans one. `pins.json` fixes all
four PCM identities and 204 complete native arrays. It does not relax the
existing five-case [diarization lane](../diarization/README.md).

## Reproduce

Use the same pinned local models, original sidecars, prepared projection/PLDA,
native Python packages and existing upstream checkout as that lane. Preparation
downloads nothing and refuses an existing output directory. Commands below run
from the repository root; model paths are the conventional locations documented
in the speaker/clustering recipes and may be supplied explicitly elsewhere.

```powershell
python -X utf8 tests/pyannote/dialogue/prepare.py --reference-source external/pyannote-audio --output artifacts/dialogue-new
python -X utf8 artifacts/dialogue-new/recipe/generate_reference.py --reference-source external/pyannote-audio --segmentation models/pyannote-segmentation/segmentation/model.onnx --embedding-directory models/pyannote-embedding --projection models/pyannote-embedding/projection.onnx --prepared-plda models/pyannote-community-1/plda.json --plda-directory models/pyannote-community-1/plda --audio-manifest artifacts/dialogue-new/assets/audio.json --output artifacts/dialogue-new/reference
dotnet build artifacts/dialogue-new/recipe/DiarizationReplay.csproj -c Release --tl:off --nologo -v minimal -o artifacts/dialogue-new/bin
dotnet build artifacts/dialogue-new/recipe/DiarizationDetail.csproj -c Release --tl:off --nologo -v minimal -o artifacts/dialogue-new/bin
dotnet artifacts/dialogue-new/bin/DiarizationReplay.dll artifacts/dialogue-new/reference models/pyannote-segmentation/segmentation/model.onnx models/pyannote-embedding/embedding_encoder.onnx models/pyannote-embedding/projection.onnx models/pyannote-community-1/plda.json artifacts/dialogue-new/public.json
dotnet artifacts/dialogue-new/bin/DiarizationDetail.dll artifacts/dialogue-new/reference models/pyannote-segmentation/segmentation/model.onnx models/pyannote-embedding/embedding_encoder.onnx models/pyannote-embedding/projection.onnx models/pyannote-community-1/plda.json artifacts/dialogue-new/detail
python -X utf8 artifacts/dialogue-new/recipe/audit.py --reference artifacts/dialogue-new/reference --public artifacts/dialogue-new/public.json --detail artifacts/dialogue-new/detail --output artifacts/dialogue-new/audit.json
```

Preparation copies the shared generator, auditor and C# runners into a separate
recipe directory with the dialogue pins. The projects point back to this source
checkout and embed the corpus/recipe identities. Source archives can use the same
preparation path without relying on an external working tree for managed code.
Keep the prepared directory immutable once references or runners are built.

The public runner makes eleven real requests: every case twice, one recovery
request after invalid/canceled inputs, and two concurrent requests on one
instance. It checks retained results and input ownership. The detail runner
saves all 176 intermediate arrays. The auditor retains the original scaled
`1e-4` numerical gate and exact decisions, with separate application and numeric
verdicts. Exit 1 with `execution_complete=true` remains a numerical failure.
Run the shared `test_refusals.py` with these paths to test malformed evidence.

Install the separate [accuracy requirements](../accuracy/requirements.txt) to
score native and managed timelines against human annotations:

```powershell
python -X utf8 tests/pyannote/dialogue/score.py --corpus artifacts/dialogue-new/assets/audio.json --reference artifacts/dialogue-new/reference/manifest.json --public artifacts/dialogue-new/public.json --output artifacts/dialogue-new/accuracy.json
```

Repeat `--public` for another configuration. The scorer verifies corpus and
reference bindings and first-request coverage, and records input/tool hashes.
It scores the entire recording with optimal one-to-one anonymous speaker
matching, zero collar and overlap included. It does not replace the complete
conformance auditor. Report ordinary and exclusive scores separately, together
with missed, false-alarm and confused speaker-seconds. No broader accuracy or
long-recording resource claim follows from this sample.

## Recorded qualification

Initial Windows replay uses SDK 10.0.204 archived product source `8e93aa7`
(core SHA256 `07ad08d790972bd69f0e3f99cd3fce5e495a3fd383c5c395d2cf5b92ba6594f4`)
on runtime 10.0.12. Default and preferred configurations each pass eleven API
requests, with exact native timelines and maximum centroid error `2.11597e-6`.
Both configurations produce identical public results and intermediate bits.
All 204 native arrays reproduce exactly. The shared generator also reproduces
all 97 arrays and cases from the previous five-case lane.

The complete trace retains **19 failed filterbank values** among 176 arrays /
10,170,633 values; maximum scaled error is `1.63600e-4`. All other stages pass
their original gates. No original-native/deterministic-tie differences occur
in these cases. Application success does not erase the numerical failures.

Native and both managed configurations have the same labeled scores:

| Recording | Ordinary DER | Exclusive DER |
|---|---:|---:|
| Full 30 seconds | 5.2074% | 10.2909% |
| Seconds 0–10 | 45.1484% | 45.1484% |
| Seconds 10–20 | 41.1523% | 45.3347% |
| Seconds 20–30 | 29.2263% | 35.2805% |

The full recording has 24.35 reference speaker-seconds. Its ordinary errors are
0.488406 seconds missed, 0.472156 false alarm and 0.307437 confusion. The full
recording forms two learned clusters; each short excerpt forms one, with some
additional count-only overlap speakers. This sample supports retaining context
across turns; it does not establish general accuracy or identify the speakers.
Initial local process peaks stay below 3.70 GB in these finite runs. AMD replay
and longer-recording resource qualification remain separate requirements.
