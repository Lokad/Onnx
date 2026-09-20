# Current core package consumption — September 20, 2026

The current `Lokad.Onnx` core package passes the repository's independent
consumer smoke test. The consumer restored the newly built package from its
private local feed, and its loaded core DLL is byte-identical to the packaged
DLL. Nothing was published.

Source `3d8664e551c0ed92fcbec1be375433fb5f483168` was archived before building.
Its production source, backend/tensor test source and checked build configuration
are unchanged from `087e280b5ea0a6a610399ccffd1a1e5668def10e`. That revision's
[existing qualification](e5/softmax-zero-product/results-20260919.md) includes
the full 3,027 backend and 342 tensor tests with the experimental softmax route
both disabled and enabled, native e5 and shared-model output checks, and AMD
contracts. Those successful suites were not rerun for this package-only check.
The experimental route remains disabled by default.

Using SDK 10.0.204 and .NET 10.0.12 on Windows, `eng/smoke-pack.ps1` successfully:

- Built the Release package and checked its required files and documentation.
- Verified exactly one runtime dependency: Google.Protobuf 3.33.5.
- Restored into a fresh private package directory and ran an independent app.
- Executed a Relu graph, imported and executed MNIST from a file and byte array,
  and checked import/execution through a sliced `ReadOnlyMemory<byte>`.

The retained copy of the script replaces only successful scratch-directory
deletion with a preservation message. Every original package and consumer check
is unchanged. The terminal script returned zero and printed all four success
markers. It suppresses underlying build stdout, so this evidence does not
provide a build-warning inventory. The complete consumer project, dependency
assets, private feed/cache, binaries and script log remain available.

Independent verification checked the source equivalence again, package XML and
contents, actual restore metadata, and identical SHA256 values for the built,
packaged, privately restored and consumer-output core DLLs. The private cache's
package bytes also match the new local feed. Global package caches were not
cleared or modified to select this package.

`Lokad.Onnx.Data` and the CLI are separate repository projects. In particular,
the core NuGet package does **not** contain the audio orchestration APIs in
Data. This check adds no audio tensor/accuracy qualification or new performance
measurement; the [support matrix](../docs/model-support.md) retains those limits.

To run the repository's ordinary package smoke on a checkout:

```powershell
pwsh -NoProfile -File eng/smoke-pack.ps1
```

For this retained run, the complete source archive and one-line retention
adaptation are under `artifacts/current-package-20260920`. Its independent
`verify.py` and closed receipt bind 2,249 files. Completed writers must not be
rerun into that artifact.

| Evidence | SHA256 |
|---|---|
| Source archive | `01cb9b72370c0e78da3df13206df6e8f1958b410116d94b53c9f88639f63deac` |
| Core package, 547,035 bytes | `2fd31cbab59eb27995a6f0b803a389cf23c2c6d1f7cbc6b9ec7c18fab431a261` |
| Core DLL, 723,968 bytes | `5c94abc1d570c0e82f07bcc483e84fd7de4207810b8b6565c70845b8fc41e821` |
| Closed package receipt | `36920b514f8ce13da9582f88402928a76f5d97f6752239602f4a5e0a578cf90e` |
| Prior complete product qualification/comparison receipt | `e654381efa0bf9d652cf9a8424357978b04d967e76ebe586458924721b98fdcb` |
