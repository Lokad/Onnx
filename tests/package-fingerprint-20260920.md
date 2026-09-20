# Current fingerprint-cache core package consumption — September 20, 2026

The current core package passes an independent consumer from a fresh private
feed/cache, both with the default settings and with
`LOKAD_ONNX_FINGERPRINT_STRINGS=1`. Both fresh processes execute Relu and MNIST
imported from a file, bytes and a sliced `ReadOnlyMemory<byte>`, with the expected
outputs. Nothing was published. This supersedes the [earlier package check](package-current-20260920.md)
for current-core packaging coverage; its earlier evidence remains unchanged.

Source `f26d939` was archived before building. Every one of its 2,316 source
files was checked against Git, explicitly accounting for 960 text files with
CRLF conversion. Production sources, backend/tensor tests and build configuration
are unchanged from archive-qualified `faf2844`. The existing
[full product qualification](e5/fingerprint-product/results-20260920.md) covers
the cache in both settings: full Windows/AMD suites and complete native e5 and
shared-model arrays. Those successful model campaigns were not repeated here.
The cache remains off by default; this package check establishes no speedup.

The repository package smoke ran with SDK10.0.204 and .NET10.0.12 on Windows.
Its retained copy preserves all original checks, retains the private app instead
of deleting it, builds explicitly with `--tl:off`, and emits the consumer output.
The generated smoke app builds with **eight nullable warnings and zero errors**.
The script suppresses core-pack build stdout, so there is no complete core-pack
warning inventory. The already built consumer then runs in a separate process
with the fingerprint-cache setting enabled; all three output markers match.

Independent verification confirms the required package contents, exactly one
runtime dependency (Google.Protobuf3.33.5), and identical core DLL bytes in the
build output, package, private cache and consumer output. The package bytes also
match the newly created feed and cached nupkg. Actual restore metadata selects
that feed for Lokad.Onnx and nuget.org for Protobuf; both private-cache DLLs match
the consumer output. Visual Studio's fallback package folder is configured, but
the actual `packagesPath` is the new private cache. No global caches were cleared.

The core package excludes `Lokad.Onnx.Data` and the CLI. Audio orchestration
therefore remains a separate repository-project capability; this is neither an
audio package qualification nor a new tensor or performance result.

The initial preparation verifier omitted `.sh`, `.ini` and `.proto` from its
text-line-ending allowance. Subsequent closure checks incorrectly assumed one
configured package folder and an audit filename that does not exist. All failed
verifier versions and corrections are retained. The corrected checks inspect
actual source bytes, restore metadata and existing product receipts. Packaging
and the two consumer executions were each performed once.

All observed process identities are terminal. The closed artifact
`artifacts/fingerprint-package-20260920` binds 2,420 files, including the source,
scripts, failed checks, private app/feed/cache, actual outputs and dependencies.
Completed writers must not be rerun.

| Evidence | SHA256 |
|---|---|
| Core package, 548,039 bytes | `a75b90f0b46304c6b3e0c537ebcd97bc138a2acb25b97e7defafb16d16080860` |
| Packaged/consumed core, 726,016 bytes | `af929e8a861185e1a8b601ff14de9beed6063f537b1db87769213709fc90c541` |
| Closed receipt | `eca97abb02aaf3e806bd814196153764dbc89d4fa7d4f1825078a532d93c06a5` |
| Referenced archived product qualification | `b2631477f8f4b227ececf50ea48fed12f68581d706d56d8dc3c4f268819718a7` |

The newly packaged DLL has different build bytes from the prior qualified DLL;
their production source equivalence is checked rather than claiming identical
build outputs. The [support matrix](../docs/model-support.md) retains the broader
e5 and audio qualification limits.
