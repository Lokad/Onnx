# Complete store-width qualification

Initial ordinary-store preparation session 10263 passed both existing local
2,882-case modes. Review identified that the old boundary grid covers tails
of one, two and seven floats, leaving four new helper branches unexercised.
No VM deployment or timing used that initial preparation.

`complete.py` appends 384 finite, zero and non-finite cases for tails three
through six, both alone and following a full 32-column panel, with and without
bias and with contiguous/offset strided destinations. The original 2,882 cases
remain the exact prefix; all 22 timing-shape records and consumer source bytes
remain unchanged. The new manifest differs only by this extra qualification.

Both expanded local modes pass **3,266 cases**. Preparation session 44623
and staging exit zero, verifying nine payload and 222 installed runtime files.
Use `C:/Python313/python.exe -X utf8 -B` from repository root with:

    tests/pyannote/direct-output-store/complete.py prepare
    tests/pyannote/direct-output-store/complete.py stage
    tests/pyannote/direct-output-store/complete.py launch
    tests/pyannote/direct-output-store/complete.py observe
    tests/pyannote/direct-output-store/complete.py collect
    tests/pyannote/direct-output-store/complete.py audit

Supervisor 669088 / birth 1790050620.12 was launched once. Observe that owner;
do not repeat preparation, staging or launch. The final report, once present,
supersedes this launch-time status.

Current artifacts: `artifacts/pyannote-direct-output-store-v2-20260922` and
`/dev/shm/lokad-pyannote-direct-output-store-v2-20260922`.
Payload SHA256: `52817780d761cea799ec7a233c7446d1ac235f728212b410c38ef8232d5d9b4f`.
No workload, numerical gate, resource bound or timing threshold was removed
or relaxed. Production and accepted application benchmarks remain unchanged.
