# Executed successor

The original preparation stopped before builds or VM writes because Python's
encoding name is `utf-8-sig`, not `utf8-sig`. The original tool and failure
receipt are preserved. `successor.py` corrects that spelling in preparation
and audit and uses fresh v2 paths. No arithmetic, validation or timing gate
changes. Use these commands, each prefixed with
`C:/Python313/python.exe -X utf8 -B` from the repository root:

    tests/pyannote/convolution-reduction/successor.py prepare
    tests/pyannote/convolution-reduction/successor.py stage
    tests/pyannote/convolution-reduction/successor.py launch
    tests/pyannote/convolution-reduction/successor.py observe
    tests/pyannote/convolution-reduction/successor.py collect
    tests/pyannote/convolution-reduction/successor.py audit

Preparation and staging have completed successfully. Local correctness covers
3,014 cases and 2,124,892 output values, with complete guards and input checks.
The exact consumer SHA256 is
`c5ea607d75e541193c87e99df760e86dd540fad0919c26311a2be7cfdda5d4cb`.
Six payload files and 222 existing remote runtime files were verified.

Artifacts are `artifacts/pyannote-convolution-reduction-v2-20260922` and
`/dev/shm/lokad-pyannote-convolution-reduction-v2-20260922` on the VM.
Supervisor PID 666152 / birth 1790047881.64 was launched once. Observe or
collect that owner; preparation, staging and launch are not polling commands.
The final result report, once present, supersedes this launch-time status.
