# Collection recovery

All four timing processes and supervisor796770/birth1790144828.17 were
terminal with code zero before collection. The SSH archive transfer completed;
local extraction then failed with `OSError: [Errno 28] No space left on device`.
The complete archive and partial extraction were preserved.

NTFS compression of two closed historical evidence directories recovered
storage without changing their contents. Every archive member was checked
against the embedded collection receipt before a fresh extraction from that
same archive. All400collected files then matched their hashes. No model
request, timing process, transfer or successful qualification was repeated.
The original failed extraction remains in `collection-partial-disk-full`.

The baseline closure includes `collection-disk-failure.json`,
`collection-recovery.json`, the complete archive and retained partial files.
ArchiveSHA256: `ee4e82e418198ecc6edfa63fbd7b489e81d23de7800c21523e3f4e421d04caf6`.
Collection receiptSHA256: `fbd86bca5f90fc464c47d69d83b3ef8427f933d4ebf9d29469357ec91c25d358`.
Baseline closureSHA256: `2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3`.

The independent audit passes all320requests,42repeatability controls and
1,881resource observations. No samples were removed or changed during recovery.
