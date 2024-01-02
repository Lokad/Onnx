using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx.Backend.Tests
{
    public class TensorTests
    {
        [Fact]
        public void CanBroadcast()
        {
            var a = new DenseTensor<int>(new[] { 256, 256, 3, });
            var b = new DenseTensor<int>(new[] { 3, 1 });
            b[0,0] = 1;
            b[1, 0] = 2;
            b[2, 0] = 3;
            var bc1 = b.BroadcastDim(1, 255);
            Assert.Equal(1, bc1[0, 204]);
            Assert.Equal(2, bc1[1, 254]);
            Assert.Equal(3, bc1[2, 164]);
        }
    }
}
