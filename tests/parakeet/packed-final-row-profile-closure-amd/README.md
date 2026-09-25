# Close the completed M78 profile without repeating inference

The original control succeeded. Its supervisor then refused phase before launch
because available memory was 17,776,640 bytes below its 11 GiB preflight limit.
That failed state and all control records were collected before resuming.

The separate resumed phase and wall workers both finished successfully. The
original capture function's final accounting save used a literal
`capture-state.json` path. It therefore overwrote that remote file with the
resumed checkpoint, and the resumed supervisor's final integrity check failed.
The original local state remains unchanged. The remote replacement is exactly
the final successful-run checkpoint before the supervisor recorded this error.
Every other originally collected file is unchanged.

`close.py` accepts only this specific failure, retains both state versions and
uses the original per-process checker with each actual owner. Neither supervisor
is relabeled successful. Five tests reject other failures, failed workers,
changed checkpoints and a fabricated successful supervisor.

From the repository root, these commands **have completed once**:

```powershell
C:/Python313/python.exe -X utf8 -B -m unittest discover -s tests/parakeet/packed-final-row-profile-closure-amd -p test_close.py
C:/Python313/python.exe -X utf8 -B tests/parakeet/packed-final-row-profile-closure-amd/close.py collect
C:/Python313/python.exe -X utf8 -B tests/parakeet/packed-final-row-profile-closure-amd/close.py audit > artifacts/parakeet-packed-final-row-profile-closure-audit-stdout-20260925.json
C:/Python313/python.exe -X utf8 -B tests/parakeet/packed-final-row-profile-closure-amd/close.py compare > artifacts/parakeet-packed-final-row-profile-closure-compare-stdout-20260925.json
```

Collected 461 files. All 240 requests, 180 measured records, public results,
1,366 resource samples, phase/node clocks and process accounting pass the
unchanged checks. Profile closure:
`df29a31afb6c5bf4bfd92f150efc4d83f3a5898695a20d1742d159db888d698d`.

Raw evidence lives in
`artifacts/parakeet-packed-final-row-profile-closure-amd-20260925`.
The original control/failure is retained in
`artifacts/parakeet-packed-final-row-profile-amd-20260925/capture-collected`.
The resumed launch/specification/observations remain in
`artifacts/parakeet-packed-final-row-profile-resume-amd-20260925`.
Do not invoke either earlier full-capture collector/auditor or rerun a worker.

[Interpretation and next experiment](../packed-final-row-profile-results/diagnosis-20260925.md).
