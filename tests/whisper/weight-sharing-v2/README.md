# Whisper sharing qualification with exact cached-name transitions

The preceding candidate failed its final all-field snapshot equality check.
A controlled diagnostic on Windows and AMD identifies two existing cached
transpose name changes in both shared and unshared graphs; every byte remains
unchanged. The earlier failed campaign remains failed.

`WeightSnapshot.cs` and `weight_transition.py` allow only those two exact,
source-verified name transitions and require every other snapshot field to remain
identical. `check_transition.py` tests each independent checker against four actual
recorded transitions and 28 damaged snapshots. `prepare.py` builds a new public
consumer against the unchanged, previously tested product, saving both full weight
snapshots before assertion so a future failure remains diagnosable.

Artifact: `artifacts/whisper-weight-sharing-v2-20260920`. Tools use
`C:/Python313/python.exe -X utf8 -B` from the repository root. The prospective new
20/80-request campaign keeps all earlier application, ownership, allocation and
resource gates. Preparation is not AMD qualification or a matched ORT timing result.
