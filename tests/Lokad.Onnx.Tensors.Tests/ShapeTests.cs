using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx.Tensors.Tests
{
    public class ShapeTests
    {
        [Fact]
        public void CanPadLeft()
        {
            var a = new DenseTensor<int>(new[] { 256, 256, 3, });
            var b = new DenseTensor<int>(new[] { 3, 1 });
            b[0, 0] = 1;
            b[1, 0] = 2;
            b[2, 0] = 3;
            var pb = b.PadLeft();
            Assert.Equal(3, pb.Rank);
            Assert.Equal(2, pb[0,1,0]);

        }

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
            Assert.Throws<IndexOutOfRangeException>(() => bc1[2, 255]);
        }
    }
}
