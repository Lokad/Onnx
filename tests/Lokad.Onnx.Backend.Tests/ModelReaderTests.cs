using Lokad.Onnx.Protobuf;

namespace Lokad.Onnx.Backend.Tests;

public class ModelReaderTests
{
    sealed class Reader : IOnnxModelReader
    {
        public int Calls;
        public string? Path;
        public ReadOnlyMemory<byte> Memory;
        public Exception? Error;

        public OnnxModel Read(string path)
        {
            Path = path;
            return Description();
        }

        public OnnxModel Read(ReadOnlyMemory<byte> data)
        {
            Memory = data;
            return Description();
        }

        OnnxModel Description()
        {
            Calls++;
            if (Error is not null) throw Error;
            return new OnnxModel
            {
                Name = "custom-reader", Opset = { [""] = 13 },
                Inputs = { new OnnxValueInfo { Name = "x", ElementType = TensorElementType.Float, Dims = [2] } },
                Outputs = { new OnnxValueInfo { Name = "y", ElementType = TensorElementType.Float, Dims = [2] } },
                Nodes = { new OnnxNode { Name = "relu", OpType = "Relu", Inputs = ["x"], Outputs = ["y"] } }
            };
        }
    }

    static void Check(ComputationalGraph graph)
    {
        Assert.True(graph.Execute(new ITensor[] { DenseTensor<float>.OfValues([-1, 2]) }, true), graph.LastErrorMessage);
        Assert.Equal(new float[] { 0, 2 }, ((Tensor<float>)graph.Outputs["y"]).ToArray());
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void FileLoadUsesChosenReaderAndPreservesPath(bool bounded)
    {
        var reader = new Reader();
        const string path = "caller-owned.model";
        var graph = bounded ? Model.Load(path, reader, 0) : Model.Load(path, reader);
        Assert.Equal(1, reader.Calls);
        Assert.Equal(path, reader.Path);
        Assert.Equal(path, graph.ModelFile);
        Check(graph);
    }

    [Theory]
    [InlineData(false)] [InlineData(true)]
    public void MemoryLoadPassesTheBorrowedSliceWithoutCopying(bool bounded)
    {
        var reader = new Reader();
        byte[] bytes = [9, 8, 1, 2, 3, 7];
        var slice = bytes.AsMemory(2, 3);
        var graph = bounded ? Model.Load(slice, reader, 0) : Model.Load(slice, reader);
        Assert.Equal(1, reader.Calls);
        Assert.True(slice.Equals(reader.Memory));
        Array.Fill(bytes, (byte)0);
        Check(graph);
    }

    [Fact]
    public void InvalidArgumentsAndBudgetAreRejectedBeforeReading()
    {
        var reader = new Reader();
        Assert.Throws<ArgumentException>(() => Model.Load("", reader));
        Assert.Throws<ArgumentNullException>(() => Model.Load((string)null!, reader));
        Assert.Throws<ArgumentNullException>(() => Model.Load("x", null!));
        Assert.Throws<ArgumentNullException>(() => Model.Load(ReadOnlyMemory<byte>.Empty, null!));
        Assert.Throws<ArgumentOutOfRangeException>(() => Model.Load("x", reader, -1));
        Assert.Throws<ArgumentOutOfRangeException>(() => Model.Load(ReadOnlyMemory<byte>.Empty, reader, -1));
        Assert.Equal(0, reader.Calls);
    }

    [Theory]
    [InlineData(false, false)] [InlineData(false, true)]
    [InlineData(true, false)] [InlineData(true, true)]
    public void ReaderFailuresRemainCallSpecificAndFatalFailuresPropagate(bool memory, bool fatal)
    {
        Exception error = fatal ? new OutOfMemoryException("reader") : new InvalidDataException("reader");
        var reader = new Reader { Error = error };
        var caught = Record.Exception(() => { if (memory) Model.Load(ReadOnlyMemory<byte>.Empty, reader); else Model.Load("x", reader); });
        Assert.Same(error, caught);
        Assert.Equal(1, reader.Calls);
        reader.Error = null;
        Check(Model.Load("retry", reader));
    }

    [Fact]
    public void ReferenceReaderPreservesFileAndSlicedMemoryModelExecution()
    {
        var path = System.IO.Path.Combine(AppContext.BaseDirectory, "models", "mnist-8.onnx");
        var reader = new ProtobufModelReader();
        var fromFile = Model.Load(path, reader);
        var bytes = File.ReadAllBytes(path);
        var padded = new byte[bytes.Length + 11];
        bytes.CopyTo(padded, 7);
        var fromMemory = Model.Load(padded.AsMemory(7, bytes.Length), reader);
        Array.Clear(padded);
        var input = DenseTensor<float>.OfShape(1, 1, 28, 28);
        input.Fill(0.5f);
        Assert.True(fromFile.Execute(new ITensor[] { input }, true), fromFile.LastErrorMessage);
        Assert.True(fromMemory.Execute(new ITensor[] { input }, true), fromMemory.LastErrorMessage);
        Assert.Equal(Assert.IsAssignableFrom<Tensor<float>>(fromFile.Outputs.Values.Single()).ToArray(),
            Assert.IsAssignableFrom<Tensor<float>>(fromMemory.Outputs.Values.Single()).ToArray());
    }
}
