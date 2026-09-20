# Current LayerNorm core package consumption — September 20, 2026

**PASS:** an independent application restored and executed the current private
`Lokad.Onnx` 0.2.0 package. Four fresh processes covered the default settings,
fingerprint cache alone, wider LayerNorm alone, and both enabled. Both switches
remain disabled by default. No package was published.

The package was built with the repository's `pack.cmd`, SDK 10.0.204, from
committed archive `7bcd4a9766115d7211c85796ce692008d42ea2ac`. All 2,500 archived
files were checked against Git blobs, allowing only exact text CRLF conversion
(1,144 files). Production, backend/tensor tests and build inputs are unchanged
from qualified `4f10e8bc70627f33d00ffb06ddccf15f43464d2a`. The
[Windows and AMD product qualification](e5/layernorm-product/results-20260920.md)
supplies the complete model/test and actual AMD wide-instruction evidence.

## Independent consumer

The retained [consumer and tools](package-layernorm/README.md) use a new local
feed and separate package cache. The application has one direct package
reference, `Lokad.Onnx` 0.2.0; the only runtime dependency is
`Google.Protobuf` 3.33.5. Restore metadata confirms the selected local package
and Protobuf from nuget.org. Built, packaged, cached and actually loaded core
DLL bytes match. Cached and loaded Protobuf bytes also match.

Each process runs exact Relu execution and MNIST import/execution from a file,
bytes and sliced `ReadOnlyMemory<byte>`, comparing all ten finite outputs and
preserving inputs/source bytes. It then checks allocating, destination and
in-place LayerNorm APIs for widths 16, 17, 384 and 1,280, with and without bias,
including a constant row. Input, scale, bias and held-output checks pass. All
settings and API variants produce identical output bits.

The independent NumPy audit reconstructs the scalar double reference for all
40,728 saved normalization values across the four processes. Maximum scaled
error is **5.86258684e-8**, below the fixed `1e-6` limit. Reflection only reads
the actual package's readonly switch values. Each consumer records .NET
10.0.12, one processor, CPU 0 affinity, loaded assembly paths/hashes and no
loaded native ORT. This Windows host has no hardware Vector512 support;
successful switch consumption does not replace the separate AMD qualification.

The package contains exactly one runtime DLL plus README, changelog, license
and icon. It excludes `Lokad.Onnx.Data`, CLI and native ORT. Audio application
APIs remain repository-project capabilities.

## Execution and evidence

Pack, restore, build and the four consumer processes all exit zero. Consumer
build output explicitly reports zero warnings and errors. Captured `pack.cmd`
stdout contains only its 31-byte banner, with empty stderr; the reason its
inner .NET diagnostics are absent is not established. Thus the retained output
does **not** establish a complete warning inventory for core packaging. Fresh
package bytes and actual consumption are independently verified.

All seven stages ran once, with child-only environment settings and CPU 0
affinity. The 69 resource samples satisfy the fixed 300-second, 4 GiB group
RSS and 1 GiB available-memory limits; maximum sampled group RSS is
456,732,672 bytes. Short consumer stages have only one or two samples, so these
are sampled observations, not absolute memory peaks. All 16 observed process
identities are terminal. Existing model campaigns were left running.

The closed artifact `artifacts/layernorm-package-20260920` binds 2,766 files.
After closure, every file, all seven tool identities, eight referenced product
evidence files and all process identities were independently reverified.

| Evidence | SHA256 |
|---|---|
| Package, 548,346 bytes | `a710bf939ca489b085ab24844269bced460a46d93098f6b5ce53359ab4812cdf` |
| Built/packaged/cached/consumed core, 726,528 bytes | `fbc275c1b91f1926103d5ddc018123de479ec7547579dff98f1ea9d430e83bfa` |
| Closed receipt | `064c618c486a45d094c67a36031444c7916ea55e902609adf92d75e920e24918` |
| Final verification | `1f28403c6baf59e27758082fbe9e277a24cfc3649dd074e670199f7c8fe212d6` |
| Referenced product qualification | `5afd88417ad896ef416df9ddfe32d8a3bb61c32332f2cd7010ffe57d1c66b2b8` |

The current package has different build bytes from the earlier qualified DLL;
production source equivalence is established separately. This qualification
covers packaging and consumption, with no new model timing or numerical claim.
