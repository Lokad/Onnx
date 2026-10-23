# Protobuf reference adapter

This nonpackable project reads ONNX files using Google.Protobuf 3.33.5 and
returns the core's neutral `OnnxModel`. The `Lokad.Onnx` NuGet package has no
runtime dependencies. Applications choose their reader and its parser version;
the reference adapter is supplied as source, like the Data companion project.

For a repository source build, reference `Lokad.Onnx.Protobuf.csproj`:

```csharp
using Lokad.Onnx;
using Lokad.Onnx.Protobuf;

var graph = Model.Load("model.onnx", new ProtobufModelReader());
// Or use the adapter's established convenience API and call-specific errors:
var other = OnnxImport.LoadOrThrow("model.onnx");
```

`IOnnxModelReader` has file-path and `ReadOnlyMemory<byte>` reads returning fresh
descriptions. File reads resolve external weights from the model directory.
Memory reads borrow their input synchronously, accept slices without copying the
encoded buffer, and return independently owned payloads. Unresolved external
weights in memory input fail explicitly. `Model.Load` consumes the description.
`OnnxImport.ParseMetadata` supports inspection without materializing weights or
reading sidecars, while still parsing the whole Protobuf message.

Package-only applications may implement `IOnnxModelReader` using their own
parser, or adapt this source against their selected core/runtime versions. No
Protobuf type crosses that interface. The adapter must itself be compatible with
its selected runtime; this design supplies no assembly-version isolation.

Before 0.4.0, the core exported `OnnxImport`, `ProtoConversions` and generated
`Onnx.*` message types. Their full names remain here, but their defining assembly
is now `Lokad.Onnx.Protobuf`. Rebuild existing clients with an adapter reference;
old binaries and old common benchmark runners cannot simply swap in the new
core. Engine/DTO APIs stay in `Lokad.Onnx`.

The adapter links the core's neutral `TensorLimits` source to share geometry
validation without granting friend access or copying maintained bounds. The
[generated schema](Import/Generated/README.md) and existing import diagnostics
are preserved. Package independence is checked by
[smoke-pack](../../eng/smoke-pack.ps1); import/external-data/ownership behavior is
checked by backend tests.
