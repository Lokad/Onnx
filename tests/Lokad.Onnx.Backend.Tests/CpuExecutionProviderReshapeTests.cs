using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests
{
    public class CpuExecutionProviderReshapeTests
    {
        [Fact]
        public void CanReshape()
        {
            var X = DenseTensor<int>.Ones(2, 3, 4);
            var s = DenseTensor<long>.OfValues(new long[] { 4, 2, 3 });
            var r = CPU.Reshape(X, s);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(r.Outputs![0].Dims, new int[3] { 4, 2, 3 });

            s = DenseTensor<long>.OfValues(new long[] { -1, 2, 3, 4 });
            r = CPU.Reshape(X, s);
            Assert.Equal(OpStatus.Success, r.Status);
            Assert.Equal(r.Outputs![0].Dims, new int[4] { 1, 2, 3, 4 });

            r = CPU.Reshape((ITensor) X, null);
            Assert.Equal(OpStatus.Failure, r.Status);
           
            Assert.Throws<ArgumentException>(() => CPU.Reshape((ITensor)X, DenseTensor<long>.OfValues(new long[,] { { 2, 2 }, { 2, 1 } })));
        }
    }
}
