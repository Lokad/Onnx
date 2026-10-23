using System.Collections;
using Lokad.Onnx.Tests.Support;

namespace Lokad.Onnx.Backend.Tests;

[Collection("SequentialLogSink")]
public class FileInputEnumerationTests
{
    sealed class OnceOnlyInputs(string[] values) : IEnumerable<string>
    {
        public int Enumerations { get; private set; }
        public int Yielded { get; private set; }
        public int Disposals { get; private set; }

        public IEnumerator<string> GetEnumerator()
        {
            if (++Enumerations != 1) throw new InvalidOperationException("Input enumerated more than once.");
            return Enumerate().GetEnumerator();
        }

        IEnumerable<string> Enumerate()
        {
            try
            {
                foreach (var value in values) { Yielded++; yield return value; }
            }
            finally { Disposals++; }
        }

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
    }

    [Theory]
    [InlineData(false, 0)]
    [InlineData(false, 1)]
    [InlineData(false, 2)]
    [InlineData(true, 0)]
    [InlineData(true, 1)]
    [InlineData(true, 2)]
    public void SinglePassSourcesPreserveOrderAndDisposeWithEitherLoggingMode(bool logging, int count)
    {
        var previous = Log.Sink;
        try
        {
            Log.Sink = logging ? (_, _) => { } : null;
            var descriptors = new[] { "mnist2.png", "mnist4.png" }.Take(count)
                .Select(name => TestSupport.CommittedImage(name) + "::mnist").ToArray();
            var stream = new OnceOnlyInputs(descriptors);
            var actual = Data.GetInputTensorsFromFileArgs(stream);
            var expected = Data.GetInputTensorsFromFileArgs(descriptors);
            Assert.NotNull(actual);
            Assert.NotNull(expected);
            Assert.Equal(count, actual.Length);
            Assert.Equal(1, stream.Enumerations);
            Assert.Equal(count, stream.Yielded);
            Assert.Equal(1, stream.Disposals);
            for (int i = 0; i < count; i++)
            {
                Assert.Equal(expected[i].Dims, actual[i].Dims);
                Assert.Equal(((Tensor<float>)expected[i]).ToArray(), ((Tensor<float>)actual[i]).ToArray());
            }
        }
        finally { Log.Sink = previous; }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void InvalidInputStopsTheSinglePassAndDisposesTheIterator(bool logging)
    {
        var previous = Log.Sink;
        try
        {
            Log.Sink = logging ? (_, _) => { } : null;
            var image = TestSupport.CommittedImage("mnist4.png") + "::mnist";
            var stream = new OnceOnlyInputs([image, "unsupported.xyz", image]);
            Assert.Null(Data.GetInputTensorsFromFileArgs(stream));
            Assert.Equal(1, stream.Enumerations);
            Assert.Equal(2, stream.Yielded);
            Assert.Equal(1, stream.Disposals);
        }
        finally { Log.Sink = previous; }
    }

    [Fact]
    public void NullInputsStillThrow() => Assert.Throws<ArgumentNullException>(() => Data.GetInputTensorsFromFileArgs(null!));
}
