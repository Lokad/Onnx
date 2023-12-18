using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx.Backend.Tests
{
    public class OpTests
    {
        [Fact]
        public void SqueezeTests()
        {

            var d = new DenseTensor<int>(new[] { 1, 2, 2, 3, 4, 4, 5, 4 });
            var sd = d.Dimensions.ToArray();
            Assert.Equal( new[] { 1, 2, 3, 4, 5, 4 }, sd);
        }
    }
}
