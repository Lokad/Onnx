# Private Whisper memory candidate: coherent build and package boundary

These tools apply exactly the five already-tested private memory changes to a
fresh current Git archive. They build the full solution, run the missing tensor
suite and verify an independent consumer of the resulting core NuGet package.
No audio model or VM workload runs here, and production source remains unchanged.

The consumer preserves the prior private-package import/normalization checks and
adds actual public released-buffer budgets 0, 31, 32 and 64 bytes. It verifies
allocation counters, exact outputs, input/held-output ownership and negative-budget
rejection in all four fingerprint/LayerNorm switch combinations.

Use `C:/Python313/python.exe -X utf8 -B` with `prepare.py`, then `run.py --artifact
artifacts/whisper-memory-product-20260921`, then `audit.py` with the same argument.
Every writer creates new evidence. The existing candidate backend proof is reused
only after exact source comparison; new build/package/consumer DLLs have their own
identities. This checks a private candidate and does not promote it or publish a
package. Separate AMD endurance and public-contract qualification remain required.
