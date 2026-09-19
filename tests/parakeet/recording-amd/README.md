# AMD Parakeet recording qualification

This finite Linux replay reuses the closed Windows recording payload and native
application reference. It runs ten sequential recording requests (including
600-second cyclic speech and silence), two concurrent silent calls, sixteen
invalid/canceled requests with recovery, and two CLI calls. It does not generate
new ORT latency measurements or qualify all intermediate numerical arrays.

The source payload and reference are pinned through the original closed receipt.
Only exact portable DLLs, required PCM files and one CLI WAV are staged. Sparse
zero storage and hard links preserve full logical file hashes. Canonical model
files are verified in place. All processes use CPU 2 and normal runtime settings;
the supervisor uses CPU 0. Prospective per-worker limits are 13 GiB sampled RSS,
1,800 seconds and at least 256 MiB available system memory.

From the repository root, run the evidence tests and prepare a new artifact:

```powershell
C:/Python313/python.exe -B -m unittest discover -s tests/parakeet/recording-amd -p test_*.py
C:/Python313/python.exe -B tests/parakeet/recording-amd/prepare.py --artifact artifacts/parakeet-recording-amd-20260919
```

Transfer the resulting tar archive to a new artifact directory on the VM, check
its hash against `preparation.json`, and extract only regular relative files.
With `<payload>` denoting that directory, use sequentially:

```text
python3 -B <payload>/remote.py install <payload>
python3 -B <payload>/remote.py launch <payload>
```

`deployment.json` records the supervisor PID and Linux start tick. Poll that
identity and `result/identity.json`; do not launch another inference process on
the VM. Once terminal, `remote.py collect <payload>` proves process absence and
writes a new compressed archive of all unique outputs and their file hashes.
Copy it locally, check its returned length/hash, safely extract to `collected`,
and save the returned JSON as `download.json`. Reusable local payload files are
also required by the final auditor:

```powershell
artifacts/asr-labeled-20260919/venv/Scripts/python.exe -B tests/parakeet/recording-amd/audit.py --artifact artifacts/parakeet-recording-amd-20260919 --output artifacts/parakeet-recording-amd-20260919/audit.json
```

The independent recording validator checks window boundaries, frame and token
contracts and partial-result semantics. Exact application results must also
match the retained native and Windows results. Auditing performs no inference.
The native reference originally crosschecked 38 windows against the upstream
decoder. Every request time and resource sample is retained; none is discarded.
All output writers refuse existing files. Do not rerun a completed campaign or
increase failed resource limits retrospectively.
