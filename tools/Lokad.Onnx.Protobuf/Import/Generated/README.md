# Generated ONNX protobuf messages

`Onnx.g.cs` is C# generated from the pinned upstream ONNX schema `onnx.proto`
and compiled into the nonpackable `Lokad.Onnx.Protobuf` adapter assembly. Never hand-edit the
generated file; change the schema pin and regenerate instead.

- Schema: `onnx/onnx` tag `v1.19.0`, commit `57b9c6a4f6eebb09ae1b34cfb632078518f72832`,
  file `onnx/onnx.proto` (proto2 syntax, `package onnx`, no `csharp_namespace`;
  protoc derives the C# namespace `Onnx` from the package name).
- Generator: `protoc` from the `Google.Protobuf.Tools` 3.33.5 NuGet package
  (build-only; never shipped), matching the `Google.Protobuf` 3.33.5 runtime
  the reference adapter owns.
- Reproduce from the repository root (PowerShell):

      protoc --proto_path=tools/Lokad.Onnx.Protobuf/Import/Generated `
        --csharp_out=tools/Lokad.Onnx.Protobuf/Import/Generated `
        --csharp_opt=file_extension=.g.cs `
        tools/Lokad.Onnx.Protobuf/Import/Generated/onnx.proto

  which writes `Onnx.g.cs` next to the schema. The output must be
  byte-identical to the checked-in file for the pinned inputs.

- License: the schema carries `SPDX-License-Identifier: Apache-2.0`
  (upstream `onnx/onnx`); the generated code derives from it.