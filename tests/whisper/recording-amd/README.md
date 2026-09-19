# Whisper recording API and CLI on AMD

This finite replay extends the existing timestamp-guided recording checks to
Linux AMD using the exact Windows-qualified portable binaries and retained native
application reference. It covers four constructed recordings and a repeat,
empty input, 600-second silence, two concurrent silent calls, ten refusal/recovery
checks, a real short-API regression and the connected-recording CLI.

It does not measure a fresh ORT latency baseline, 600-second speech, or all
intermediate tensors. Confidence floats are diagnostics; discrete transcript,
token, timestamp, seek, stop and no-speech decisions must match. The independent
validator reconstructs every committed segment from actual tokens and checks
each engine's no-speech decision against its own confidence values.

Workers inherit CPU 2 before startup and use normal runtime settings; the
supervisor uses CPU 0. Prospective per-worker guards are 13.5 GiB sampled
process-group RSS, 1,800 seconds and at least 256 MiB available system memory.
Every output and half-second resource sample is retained, including failures.

From the repository root, with a new output directory:

```powershell
artifacts/asr-labeled-20260919/venv/Scripts/python.exe -B -m unittest discover -s tests/whisper/recording-amd -p test_*.py
C:/Python313/python.exe -B tests/whisper/recording-amd/prepare.py --artifact artifacts/whisper-recording-amd-20260919
```

Preparation verifies the closed recording receipt and frozen binaries. It copies
the existing Linux process helper unchanged into the payload; the prior Parakeet
source is not edited. The compressed transfer omits existing connected/shifted
PCM and connected.wav. Installation checks their complete file hashes and the
closed Parakeet collection before creating hard links in the new directory.
All fourteen canonical model assets are verified in place; no weights transfer.

After safe extraction of the digest-verified archive to `<payload>` on the VM:

```text
python3 -B <payload>/remote.py install <payload>
python3 -B <payload>/remote.py launch <payload>
```

Observe the exact PID/start identity in `deployment.json`. Once the supervisor
and both workers are terminal, `remote.py collect <payload>` verifies absence of
every sampled process identity and group before writing an archive and inventory.
Verify the returned archive length/hash locally, extract only regular relative
files into `collected`, and save the returned metadata in `download.json`.

```powershell
artifacts/asr-labeled-20260919/venv/Scripts/python.exe -B tests/whisper/recording-amd/audit.py --artifact artifacts/whisper-recording-amd-20260919 --output artifacts/whisper-recording-amd-20260919/audit.json
```

All successful writers refuse existing destinations. Do not repeat a failed
run with relaxed criteria. The [Windows recording report](../recording/results-20260919.md)
and [matched short-clip ORT benchmark](../../audio/whisper-comparison/results-20260919.md)
retain their distinct scopes and known numerical limitations.
