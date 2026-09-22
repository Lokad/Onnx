# M22 Parakeet regression on AMD

Following complete Pyannote qualification, compare measured selected Core
`3c2f16b0` / Data `6318cf48` and candidate Core `208371f6` / Data `b9358370`.
Both use the unchanged TranscribeReplay and AudioBenchmark binaries. No product
or consumer is rebuilt, and existing pinned models are reused on the VM.

From repository root use `C:/Python313/python.exe -X utf8 -B` with
`selftest.py`, then `run.py prepare`, `run.py stage`, `run.py launch`.
Use `run.py observe`; after terminal state, `run.py collect` and `audit.py`.
Existing destinations and changed inputs are refused.

Four sequential workers run selected native/public, then candidate native/public.
Each native replay covers 784 arrays / 3,090,494 values, seven cases and six
argument rejections. Require every original Microsoft ORT scaled-error check at
1e-4, then exact candidate/selected output bytes and complete transcription
objects. Twenty public clips preserve exact text, tokens, timestamps, readonly
PCM and retained outputs. Original independent auditors are copied byte for byte.
The three historical Windows native failures are not permitted in this AMD gate.

All original request clocks are retained, with no performance score. This is
shared-runtime regression coverage for the Pyannote candidate; Parakeet
optimization remains second in priority.

CPU2 affinity precedes CLR startup; monitoring runs on CPU0. Workers allow no
LOKAD/DOTNET/COMPlus overrides. Before execution freeze 12 GiB available / 3 GiB
tmpfs preflight, 12 GiB maximum owned RSS, 1,800 seconds per worker, 1 GiB minimum
available memory/tmpfs, 1 GiB output and 2 GiB total artifacts. These prospectively
declared full-model bounds replace only this new phase's component bounds.
Collection streams to the local machine after all recorded owners terminate.
