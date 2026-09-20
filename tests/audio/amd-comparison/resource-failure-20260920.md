# AMD audio comparison stopped during Whisper conformance

The all-family campaign failed its unchanged available-memory guard before any
timing process began. Managed Whisper completed 16 of 20 requests; all sixteen
saved token/text/stop and input/ownership checks pass. This prefix does not qualify
the complete corpus, final held-output check or repeated-process memory behavior.

The last sample has **1,027,723,264 bytes available**, below the
**1,073,741,824-byte reserve**, with **14,977,687,552 bytes RSS**.
The separate RSS, disk, time and CPU-affinity guards still pass. All preceding
978 samples pass every guard. The supervisor terminated its owned managed worker;
all seven original process identities are absent. The complete 979-sample sequence,
partial outputs and traceback remain intact. This is a resource failure, not a
Microsoft ORT timing comparison or a numerical mismatch.

Parakeet and pyannote both complete conformance under both engines: 48 calls in
four workers. Native Whisper also completes all twenty calls and the complete
Linux feature-array checks. Every complete worker's identity, request, output,
resource sample and original raw record was independently checked. No worker was
restarted, no tolerance changed and no timing number is inferred from conformance.

All sixteen saved managed Whisper calls allocate approximately 597–718 MB each.
After the first call, the generation-2 collection count stays at 20 through the
last saved call; younger-generation collections continue. These observations
support investigating allocation and collection behavior. They do not establish
which objects remain reachable or prove a leak. Neither forced collection nor a
larger memory limit is applied to turn this failed run into a success.

Hardware is AMD EPYC 9V74, CPU 2, .NET 10.0.8 / ORT 1.29.0, product source
`1d10d22`. Exact private and native-library identities remain in the original
frozen manifest, SHA-256 `fc9f53cb64d7bc2e61ad637c84629114d56152f4fe212d4e9f010aeba7f6f6d9`.
The collection binds 242 files and verifies
18,821 external model/library files. Archive SHA-256:
`205348a8274ac2d48fda6d279be3ce8a7c127c704dbc3c649dd6dd71dbfb91ca`.

A separate prospective Parakeet/pyannote timing lane may reuse their four complete
checks after verifying unchanged model, consumer, runtime and product identities.
It must retain this all-family failure and describe Whisper's AMD timing as pending.
The historical Windows comparison remains valid within its stated scope.
