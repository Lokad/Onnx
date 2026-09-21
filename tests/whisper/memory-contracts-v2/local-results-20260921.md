# Whisper memory candidate: Windows public request contracts

The private memory candidate passes **13 completed requests and 16 refusal or
cancellation checks** on Windows, .NET 10.0.12, logical CPU 2. The two concurrent
non-silent requests use the same transcriber and their measured request lifetimes
overlap. Both retain exact native text, tokens, stop and no-speech decisions.
After invalid and canceled requests, a successful short request reproduces the
earlier complete managed result exactly, including confidence fields.

| Recording | Windows | Segments | Stop reason | Exact native decisions |
|---|---:|---:|---|---|
| connected | 3 | 11 | Completed | Yes |
| shifted | 3 | 10 | Completed | Yes |
| token-limit | 1 | 0 | TokenLimit | Yes |
| window-limit | 1 | 2 | WindowLimit | Yes |
| connected (repeat) | 3 | 11 | Completed | Yes |

The remaining requests cover empty input, ten-minute digital silence, two
concurrent silent recordings, the existing short regression, two overlapping
speech calls and one short recovery. The finite connected and shifted recordings
do not establish arbitrary-duration speech behavior. The original repeated
recording also retains its complete managed result exactly.

All actual prior output objects and PCM inputs remain held and unchanged. The
independent auditor reconstructs recording segments and seek decisions from the
tokens and compares every public decision with the pinned native references.
Native confidence differences remain reported diagnostics. Scheduling cancellation
after 50 ms proves the observed cancellation and recovery; it does not locate the
precise neural instruction at which the token became canceled.

## Weight and resource checks

The complete before/after decoder snapshots preserve every initializer payload,
shape, type, membership, graph binding, packing total and shared-storage count.
Only the two independently verified cached transpose tensor names acquire their
graph-output names. All **1,136** comparisons of
original serialized initializer hashes, shapes and types pass. The candidate
reports and independently checks **635,187,200 shared initializer bytes**.

All **781** resource samples pass the 1,800-second, 14 GiB RSS,
1 GiB available-memory and 32 MiB disk limits. Peak sampled RSS is
**12,215,046,144 bytes** and minimum available memory is
**4,563,976,192 bytes**. All actual process identities are terminal.
No forced collection or LOKAD/DOTNET/COMPlus override is used. These observations
are not a before/after RSS comparison or matched Microsoft ORT latency.

The diagnostic build has zero warnings/errors; the underlying unchanged private
product previously passed 3,101 backend tests, including twelve weight-sharing
cases, with 93 hardware skips. Its DLLs match those actually exercised by the test
process. The completed-result audit rejects **12**
damaged application and weight records.

## Scope and evidence

The first contract attempt was refused at its local memory/disk preflight before
any child or inference launched. Its unused frozen artifact is preserved under
`whisper-memory-contracts-20260920`. This separate corrected consumer saves both
snapshots before validation and uses the source-proven two-name policy. The
[earlier sharing campaign failure](../weight-sharing/failure-20260920.md) and
[controlled decoder diagnosis](../weight-metadata/results-20260920.md) remain
separate evidence.

This is local qualification of a private candidate. AMD contract coverage,
coherent production integration, package checks and matched AMD Whisper timing
remain pending. Existing encoder/logit numerical failures are unchanged.

Frozen runtime/consumer receipt SHA-256: `4f5bdbb14521bcd79dae836766c2588f680074561a5875e2ee7a067f81234bbd`.
[Complete observations](local-observations-20260921.json) retain each recording
decision summary, concurrency overlap, resource totals and result identities.
Full source, inputs, native references, actual output objects' serialized results,
weight snapshots and resource samples are under
`artifacts/whisper-memory-contracts-v2-20260921`.
