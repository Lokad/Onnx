# Protobuf import implementation

ONNX loading is included in the core NuGet package. Google.Protobuf and the generated schema code under Generated/ parse .onnx files and buffers into OnnxModel descriptions for Model.Load. CLI, Data and package consumers use the same public API. See the [core guide](../README.md).

## Entry points (OnnxImport)

- Parse(path) / Parse(bytes): protobuf parse plus external-data resolution against the model file directory (missing location/file or out-of-range offset throws).
- ParseMetadata(path): parses the complete protobuf, but keeps initializer and tensor-attribute payloads empty and never reads sidecars. Tensor attributes are OnnxTensor descriptors in this mode, including in nested graphs; full Parse returns executable ITensor attributes.
- Load(path) / Load(bytes): parse plus Model.Load, with the real ModelFile stamped on the graph; returns null on nonfatal failures with the cause on LastErrorMessage/LastErrorCause plus the error log, while fatal runtime failures propagate.
- LoadOrThrow(path): direct file loading with exceptions on failure. The Load/Parse buffer overloads accept byte arrays or read-only memory; they cannot resolve external sidecars and do not retain the caller's buffer.

## Conversions (ProtoConversions)

- TensorProto to OnnxTensor (dims + typed data array), ValueInfoProto to OnnxValueInfo (same shaped-tensor contracts as before), AttributeProto to plain values, NodeProto to OnnxNode.
- The ToTensor / TensorNameDesc extension overloads on protos are thin shims over the core DTO materializers, so tests keep their existing assertions byte for byte.

