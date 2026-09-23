# Current Parakeet diagnostic consumer

Build SampledAudio against unchanged Core521bae17/Dataf3b9aa81, SDK10.0.204.
Adapt the qualified AMD consumer: Parakeet family guard and two non-inlined
public-call markers, WarmupParakeet and FullParakeet. Preserve all normalization,
native/public, input, ownership, clocks and collection barriers. The inherited
Lokad-Pyannote-Diagnostic EventSource name is retained as internal transport.
All160old methods are inventoried:159exact, Main changed, only two wrappers added.
The complete source diff and changed IL must be reviewed before capture.

Run C:/Python313/python.exe -X utf8 -B with run.py prepare,stage,launch,observe,
collect as separate commands, then audit.py. Build only the consumer on AMD;
do not rebuild product DLLs. Four jobs retain the existing12GiBavailable/3GiBtmpfs
preflight,8GiB RSS,900seconds and all resource/identity checks. Preserve failures;
refuse existing artifacts. No performance or product change follows from build.
