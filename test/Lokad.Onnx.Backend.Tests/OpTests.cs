using System.Runtime.Versioning;

namespace Lokad.Onnx.Backend.Tests
{
    [RequiresPreviewFeatures]
    public class OpTests
    {
        [Fact]
        public void CanSqueeze()
        {

            var d = new DenseTensor<int>(new[] { 1, 2, 2, 3, 4, 4, 5, 4 });
            var sd = d.Dimensions.ToArray();
            Assert.Equal( new[] { 1, 2, 3, 4, 5, 4 }, sd);
        }

        [Fact]
        public void CanBroadcast() 
        {
            var a = new DenseTensor<int>(new[] { 256, 256, 3, });
            var b = new DenseTensor<int>(new[] { 3 });
            //var c = a.Add(b);
            var r = CPUExecutionProvider.Broadcast(a, b);
            Assert.NotNull(r.Outputs);
            var ba = (Tensor<int>) r.Outputs[0];
            var bb = (Tensor<int>)r.Outputs[1];
            Assert.Equal(OpStatus.Success, r.Status);

            var c = new DenseTensor<int>(new[] { 22, 3 });
            r = CPUExecutionProvider.Broadcast(a, c);
            Assert.Equal(OpStatus.Failure, r.Status);
            r = CPUExecutionProvider.Broadcast(a, new DenseTensor<int>(new[] { 256, 3 }));
            Assert.Equal(OpStatus.Success, r.Status);
            r = CPUExecutionProvider.Broadcast(a, new DenseTensor<int>(new[] { 1, 256, 3 }));
            Assert.Equal(OpStatus.Success, r.Status);
            r = CPUExecutionProvider.Broadcast(a, new DenseTensor<int>(new[] { 256, 1 }));
            Assert.Equal(OpStatus.Success, r.Status);
        }


        
    }
}
