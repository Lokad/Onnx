extern alias OnnxSharp;

using OnnxSharp::Onnx;

namespace Lokad.Onnx.Backend.Tests;

public class ExternalDataTests
{
    static void WithTempDirectory(Action<string> exercise)
    {
        var tempDirectory = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName());
        Directory.CreateDirectory(tempDirectory);
        try
        {
            exercise(tempDirectory);
        }
        finally
        {
            Directory.Delete(tempDirectory, true);
        }
    }

    static TensorProto ExternalProto(string fileName)
    {
        var proto = new TensorProto { Name = "weight", DataType = (int)TensorElementType.Float };
        proto.Dims.Add(2);
        proto.DataLocation = TensorProto.Types.DataLocation.External;
        proto.ExternalData.Add(new StringStringEntryProto { Key = "location", Value = fileName });
        return proto;
    }

    [Fact]
    public void ResolveExternalData_Loads_Offset_Length_Slice()
    {
        WithTempDirectory(directory =>
        {
            File.WriteAllBytes(Path.Combine(directory, "weights.bin"), new byte[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
            var proto = ExternalProto("weights.bin");
            proto.ExternalData.Add(new StringStringEntryProto { Key = "offset", Value = "4" });
            proto.ExternalData.Add(new StringStringEntryProto { Key = "length", Value = "8" });
            proto.ResolveExternalData(directory);
            Assert.Equal(TensorProto.Types.DataLocation.Default, proto.DataLocation);
            Assert.Equal(new byte[] { 4, 5, 6, 7, 8, 9, 10, 11 }, proto.RawData.ToByteArray());
        });
    }

    [Fact]
    public void ResolveExternalData_Loads_Whole_File_When_Length_Missing()
    {
        WithTempDirectory(directory =>
        {
            var payload = new byte[] { 20, 21, 22, 23 };
            File.WriteAllBytes(Path.Combine(directory, "whole.bin"), payload);
            var proto = ExternalProto("whole.bin");
            proto.ResolveExternalData(directory);
            Assert.Equal(payload, proto.RawData.ToByteArray());
        });
    }

    [Fact]
    public void ResolveExternalData_Resolves_Two_Files_Independently()
    {
        WithTempDirectory(directory =>
        {
            File.WriteAllBytes(Path.Combine(directory, "first.bin"), new byte[] { 1, 1, 1, 1 });
            File.WriteAllBytes(Path.Combine(directory, "second.bin"), new byte[] { 2, 2, 2, 2 });
            var first = ExternalProto("first.bin");
            var second = ExternalProto("second.bin");
            first.ResolveExternalData(directory);
            second.ResolveExternalData(directory);
            Assert.Equal(new byte[] { 1, 1, 1, 1 }, first.RawData.ToByteArray());
            Assert.Equal(new byte[] { 2, 2, 2, 2 }, second.RawData.ToByteArray());
        });
    }

    [Fact]
    public void ResolveExternalData_Missing_File_Throws()
    {
        WithTempDirectory(directory =>
        {
            var proto = ExternalProto("absent.bin");
            Assert.Throws<FileNotFoundException>(() => proto.ResolveExternalData(directory));
        });
    }

    [Fact]
    public void ResolveExternalData_Range_Past_End_Throws()
    {
        WithTempDirectory(directory =>
        {
            File.WriteAllBytes(Path.Combine(directory, "short.bin"), new byte[16]);
            var proto = ExternalProto("short.bin");
            proto.ExternalData.Add(new StringStringEntryProto { Key = "offset", Value = "12" });
            proto.ExternalData.Add(new StringStringEntryProto { Key = "length", Value = "8" });
            Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
        });
    }

    [Fact]
    public void ResolveExternalData_Missing_Location_Throws()
    {
        WithTempDirectory(directory =>
        {
            var proto = new TensorProto { Name = "orphan", DataType = (int)TensorElementType.Float };
            proto.DataLocation = TensorProto.Types.DataLocation.External;
            proto.ExternalData.Add(new StringStringEntryProto { Key = "offset", Value = "0" });
            Assert.Throws<InvalidOperationException>(() => proto.ResolveExternalData(directory));
        });
    }

    [Fact]
    public void ResolveExternalData_Leaves_Embedded_Data_Untouched()
    {
        var proto = new TensorProto { Name = "embedded", DataType = (int)TensorElementType.Float };
        proto.Dims.Add(1);
        proto.FloatData.Add(1.5f);
        proto.ResolveExternalData(Path.GetTempPath());
        Assert.Equal(TensorProto.Types.DataLocation.Default, proto.DataLocation);
        Assert.Equal(0, proto.RawData.Length);
        Assert.Equal(1.5f, proto.FloatData[0], 5);
    }
}
