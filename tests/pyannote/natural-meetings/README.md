# Natural ten-minute meeting evaluation

This lane adds uninterrupted, human-annotated conversations to the existing
constructed long-request qualification. Results are pending. It changes no
production code, model parameter, speaker-count policy or numerical tolerance.

Selection was fixed before inference: first session-a in the official AMI test
list at each of the ES and IS sites, at setup revision
`67c2d539286e89f68952d5dcf83912bd9f01dfae`. These are ES2004a and IS1009a.
Use the first 600 seconds of each original mono 16-kHz mixed-headset WAV,
preserving the original PCM16 samples without resampling or normalization.
Both excerpts have four annotated speakers; their labeled overlapping speech
occupies 52.71 and 55.47 seconds, respectively. They are two meeting excerpts,
not a representative estimate for the whole corpus or independent utterances.

Audio/transcription source and CC BY 4.0 license:
[AMI corpus](https://groups.inf.ed.ac.uk/ami/download/).
Words-only human references and complete evaluation regions come from
[the pinned pyannote/BUT Speech@FIT setup](https://github.com/pyannote/AMI-diarization-setup/tree/67c2d539286e89f68952d5dcf83912bd9f01dfae).
References derive from manual AMI 1.6.2 annotations. The setup code uses Apache
2.0. Cite Carletta et al., *The AMI meeting corpus: A pre-announcement* (2006),
and Landini et al., *Bayesian HMM clustering of x-vector sequences (VBx) in
speaker diarization* (2022), as requested by the source setup.

Each engine runs both complete excerpts and then the first thirty seconds of
ES2004a for recovery. Inputs and both held long outputs must remain unchanged.
Managed execution uses the already qualified Core087e280/Dataf568132 DLLs on
AMD EPYC 9V74 CPU2. Native execution uses the existing complete ORT1.29.0-backed
adapter with pinned Torch/NumPy/SciPy/upstream pyannote methods on Windows
i7-14700KF CPU2. The platforms differ; one-pass API times are accuracy-replay
observations and do not replace the matched audio latency benchmarks.

Compare exact status, window counts, speaker IDs and ordered public timelines,
with the existing 1e-12 endpoint and scaled 1e-4 centroid bounds. Retain every
disagreement. Score ordinary and exclusive timelines separately using official
pyannote.metrics4.1, zero collar, overlap included, optimal speaker mapping and
the complete six hundred seconds. Report all error components, per-meeting DER
and summed-component aggregate. The recovery crop is not an extra accuracy
sample. Equal human scores cannot override failed native compatibility checks;
public output agreement does not resolve existing intermediate tensor failures.

`prepare.py --artifact <new-directory>` downloads original sources and produces
byte-exact crops, independently checked with Python wave and SoundFile.
`prepare_runtime.py --artifact <directory>` binds existing model/source pins and
qualified products. Build `NaturalMeetings.csproj` with
`-p:FrozenProductDirectory=<qualified-bin> --tl:off --nologo -v minimal`.
`supervise.py run --artifact <directory> --engine managed|native --mode inputs`
validates input decoding without inference. `--label inputs-v2` names a new
diagnostic output after a preserved failed input-only attempt.

Before inference, freeze the exact input/source/binary/scoring payload and plan.
`supervise.py launch --artifact <directory> --engine managed|native` starts one
owned worker; never relaunch on an observation timeout. It enforces 8-GiB group
RSS, 3,600 seconds, 1-GiB available memory and CPU2 affinity, sampling every half
second. Both input-only attempts initially failed before spawning workers
because psutil on Windows requires a string path. The corrected preflight and
both independent input checks pass; those failed records remain preserved.

Run the ten policy/output/resource/collection/scoring tests with the existing psutil7 environment:
`python -X utf8 -B -m unittest discover -s tests/pyannote/natural-meetings`.
After both engines finish and all recorded process births are absent, safely
collect results and run `audit.py --artifact <directory> --output <new-json>` in
the pinned metric environment. Its strict comparison and human-label scores
are separate fields. Successful writers are single-use.

The execution payload is frozen at source `00e5ad7`, manifest SHA256
`7de4bac72c5b6ac8788109322e020441db3f4f06657342917fbc38aaee01a1a9`.
The first freeze's installed-package path assumption failed before any worker
was created; the failed runtime snapshot is retained. Resolving the actual ORT
package location fixes preparation without changing inference. The native reference has completed all three calls and its processes are
terminal. The AMD managed worker is running; the comparison and human-label
scores are pending.

`collect.py --artifact <directory>` requires terminal AMD processes and verifies
all transferred files, preserving any failed run. `audit_resources.py --artifact
<directory> --output <new-json>` independently checks raw request records,
resource samples, process accounting, frozen native binaries and terminal
identities. Run both resource and application/metric audits before closure.

`close.py --artifact <directory>` rechecks original inputs, frozen files, complete
collection, raw output/resource records and current process births on both hosts.
It independently reproduces all eight human-score rows with an interval sweep
and exact speaker assignment. Damaged copies of each engine's real records must
be refused: nineteen application mutations and fourteen resource mutations.
The closure receipt preserves the full artifact and copies of postprocessing
sources. A valid completed run can close with failed public compatibility;
disagreement remains explicit and is never changed into a pass.

After closure, `report.py --artifact <directory>` writes the complete Markdown
report and JSON observations beside these tools. It includes both engines'
public outputs, every compatibility mismatch, per-meeting human error components,
summed-component aggregates, descriptive times and process resources. Neither
writer repeats inference. Both refuse existing outputs, and the closed artifact
must remain unchanged.
