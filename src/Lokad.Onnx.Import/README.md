# Lokad.Onnx.Import

ONNX protobuf loading for tooling: parses .onnx files and buffers (OnnxSharp) into the core OnnxModel DTO and hands it to Model.Load. Referenced by the CLI, the test runners and the test suite only; never packed into the Lokad.Onnx NuGet, which stays free of the OnnxSharp dependency.

## Entry points (OnnxImport)

- Parse(path) / Parse(bytes): protobuf parse plus external-data resolution against the model file directory (missing location/file or out-of-range offset throws).
- Load(path) / Load(bytes): parse plus Model.Load, with the real ModelFile stamped on the graph; returns null on nonfatal failures with the cause on LastErrorMessage/LastErrorCause plus the error log, while fatal runtime failures propagate.

## Conversions (ProtoConversions)

- TensorProto to OnnxTensor (dims + typed data array), ValueInfoProto to OnnxValueInfo (same shaped-tensor contracts as before), AttributeProto to plain values, NodeProto to OnnxNode.
- The ToTensor / TensorNameDesc extension overloads on protos are thin shims over the core DTO materializers, so tests keep their existing assertions byte for byte.

