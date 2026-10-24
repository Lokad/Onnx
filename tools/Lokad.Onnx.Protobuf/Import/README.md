# Protobuf import implementation

ONNX protobuf loading in the nonpackable reference adapter: parses .onnx files and buffers (Google.Protobuf with generated schema code under Generated/) into the core OnnxModel DTO and hands it to Model.Load. Used by CLI, Data, source consumers and tests. The core NuGet has no parser dependency. See the [adapter guide](../README.md).

## Entry points (OnnxImport)

- Parse(path) / Parse(bytes): protobuf parse plus external-data resolution against the model file directory (missing location/file or out-of-range offset throws).
- ParseMetadata(path): parses the complete protobuf, but keeps initializer and tensor-attribute payloads empty and never reads sidecars. Tensor attributes are OnnxTensor descriptors in this mode, including in nested graphs; full Parse returns executable ITensor attributes.
- Load(path) / Load(bytes): parse plus Model.Load, with the real ModelFile stamped on the graph; returns null on nonfatal failures with the cause on LastErrorMessage/LastErrorCause plus the error log, while fatal runtime failures propagate.

## Conversions (ProtoConversions)

- TensorProto to OnnxTensor (dims + typed data array), ValueInfoProto to OnnxValueInfo (same shaped-tensor contracts as before), AttributeProto to plain values, NodeProto to OnnxNode.
- The ToTensor / TensorNameDesc extension overloads on protos are thin shims over the core DTO materializers, so tests keep their existing assertions byte for byte.

