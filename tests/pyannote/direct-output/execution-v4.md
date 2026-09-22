# Explicit bias NaN selection on both platforms

The v3 AMD normal-path qualification exposed the same bias-payload mismatch
that the earlier Windows forced-scalar check found: expected `7fc12345`,
actual `ffc00000`, rows 32 / reduction 64 / columns 1, bias present.
The AMD owner and target are terminal. Collection verifies ten evidence files,
unchanged inputs and all 54 resource samples; peak RSS is 245,694,464 bytes.
Failure closure is
`c07efe05c37a1a30653d2a2c3f9fe7d1a82d255138f1b80c2a3d516f5d66c997`.
No timing process started in v3, and no performance conclusion follows.

The v4 generator makes the selected epilogue's behavior explicit: when bias
is NaN, quiet it with an addition to zero and retain that payload; otherwise
perform the ordinary post-reduction addition. This applies to both SIMD and
scalar stores and avoids depending on register allocation's operand ordering.
The original 2,882 cases and every numerical/timing gate are retained.
Normal and forced-scalar Windows qualification both pass. Product DLL bytes
remain unchanged; only isolated probe assemblies are rebuilt.

Use `C:/Python313/python.exe -X utf8 -B` from repository root with:

    tests/pyannote/direct-output/complete_v4.py prepare
    tests/pyannote/direct-output/complete_v4.py stage
    tests/pyannote/direct-output/complete_v4.py launch
    tests/pyannote/direct-output/complete_v4.py observe
    tests/pyannote/direct-output/complete_v4.py collect
    tests/pyannote/direct-output/complete_v4.py audit

Preparation session 30496 and staging exit zero. Supervisor
668034 / birth 1790049774.53 was launched once. Observe that owner; do not
repeat preparation, staging or launch. The final result report supersedes
this launch-time state.

Artifacts: `artifacts/pyannote-direct-output-v4-20260922` and remote
`/dev/shm/lokad-pyannote-direct-output-v4-20260922`.
Payload SHA256: `83e69eb661b9554913b63b7cb7fc775e0a4ad716f32901954abdd0c02ef97076`.
Earlier preparation and AMD failure evidence stays frozen in its original paths.
