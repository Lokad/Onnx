using CPU = Lokad.Onnx.CPUExecutionProvider;

namespace Lokad.Onnx.Backend.Tests
{
    public class OpTests
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

        [Fact]
        public void CanBroadcast() 
        {
            var a = new DenseTensor<int>(new[] { 256, 256, 3, });
            var b = new DenseTensor<int>(new[] { 3 });
            //var c = a.Add(b);
            var r = Tensor<int>.Broadcast(a, b);
            Assert.NotNull(r);
            var ba = (Tensor<int>) r[0];
            var bb = (Tensor<int>)r[1];
            Assert.NotEmpty(r);

            var c = new DenseTensor<int>(new[] { 22, 3 });
            r = Tensor<int>.Broadcast(a, c);
           
            r = Tensor<int>.Broadcast(a, new DenseTensor<int>(new[] { 256, 3 }));
            Assert.NotEmpty(r);
            r = Tensor<int>.Broadcast(a, new DenseTensor<int>(new[] { 1, 256, 3 }));
            Assert.NotEmpty(r);
            r = Tensor<int>.Broadcast(a, new DenseTensor<int>(new[] { 256, 1 }));
            Assert.NotEmpty(r);
        }

        [Fact]
        public void CanGetShape()
        {
            var t = DenseTensor<float>.OfShape(3, 4, 5);
            var o = CPU.Shape(t, 1);
        }
        
    }
}
