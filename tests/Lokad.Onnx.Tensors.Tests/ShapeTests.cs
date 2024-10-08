﻿using System;
using System.Collections.Generic;
using System.Linq;
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
        public void CanBroadcastShape()
        {
            Assert.True(Tensor<int>.BroadcastShape(new int[] { 1, 1 }, new int[] { 2 }, out var b));
            Assert.Equal(b, new int[] { 1, 2 });

            Assert.True(Tensor<int>.BroadcastShape(new int[] { 4, 1, 1 }, new int[] { 2 }, out b));
            Assert.Equal(b, new int[] { 4, 1, 2 });

            Assert.True(Tensor<int>.BroadcastShape(new int[] { 1, 2 }, new int[] { 2 }, out b));
            Assert.Equal(b, new int[] { 1, 2 });

            Assert.False(Tensor<int>.BroadcastShape(new int[] { 1, 3 }, new int[] { 2 }, out b));
            Assert.False(Tensor<int>.BroadcastShape(new int[] { 4, 3, 3 }, new int[] { 2 }, out b));

            Assert.True(Tensor<int>.BroadcastShape(new int[] { 2, 1 }, new int[] { 2 }, out b));
            Assert.Equal(b.Append(5).Append(6), new int[] { 2, 2, 5, 6 });

            Assert.True(Tensor<int>.BroadcastShape(new int[] { 2, 1 }, new int[] { 2, 3 }, out b));
            Assert.Equal(b.Append(5).Append(6), new int[] { 2, 3, 5, 6 });
        }
        [Fact]
        public void CanBroadcast()
        {
            Tensor<int> a = new DenseTensor<int>(new[] { 256, 256, 3, });
            Tensor<int> b = new DenseTensor<int>(new[] { 3, 1 });
            b[0,0] = 1;
            b[1, 0] = 2;
            b[2, 0] = 3;
            var bc1 = b.BroadcastDim(1, 255);
            Assert.Equal(1, bc1[0, 204]);
            Assert.Equal(2, bc1[1, 254]);
            Assert.Equal(3, bc1[2, 164]);
            Assert.Throws<ArgumentOutOfRangeException>(() => bc1[3, 256]);

            var ba = Tensor<int>.Broadcast(Tensor<int>.Ones(1, 2), Tensor<int>.Ones(3, 1));
            Assert.Equal(2, ba.Length);
        }

        [Fact]
        public void CanIterateDims()
        {
            var a = new DenseTensor<int>(new[] { 256, 212, 3, });
            var di = a.GetDimensionsIterator(0..^1);
            di = a.GetDimensionsIterator();
            while (di.Next() != null)
            {
                var i = di.Index;
            }

        }

        [Fact]
        public void CanReshape()
        {
            var X = DenseTensor<int>.Ones(2, 3, 4);
            
            var s = DenseTensor<long>.OfValues(new long[] {4,2,3});
            var Y = Tensor<int>.Reshape(X, s);
            Assert.Equal(new int[] {4,2,3}, Y.Dimensions.ToArray());

            s = DenseTensor<long>.OfValues(new long[] { 2, 4, 3 });
            Y = Tensor<int>.Reshape(X, s);
            Assert.Equal(new int[] { 2, 4, 3 }, Y.Dimensions.ToArray());

            s = DenseTensor<long>.OfValues(new long[] { 2, 12 });
            Y = Tensor<int>.Reshape(X, s);
            Assert.Equal(new int[] { 2, 12 }, Y.Dimensions.ToArray());

            s = DenseTensor<long>.OfValues(new long[] { 2,  -1, 2});
            Y = Tensor<int>.Reshape(X, s);
            Assert.Equal(new int[] { 2, 6, 2 }, Y.Dimensions.ToArray());

            s = DenseTensor<long>.OfValues(new long[] { -1, 2, 3, 4 });
            Y = Tensor<int>.Reshape(X, s);
            Assert.Equal(new int[] { 1, 2, 3, 4 }, Y.Dimensions.ToArray());

            s = DenseTensor<long>.OfValues(new long[] { 2, 0, 1, -1 });
            Y = Tensor<int>.Reshape(X, s);
            Assert.Equal(new int[] { 2, 3, 1, 4 }, Y.Dimensions.ToArray());
        }

        [Fact]
        public void CanTranspose()
        {
            var X = DenseTensor<int>.Ones(2, 3, 4);
            var tX = Tensor<int>.Transpose(X);
            Assert.Equal(2, tX.Dimensions[2]);
            tX = Tensor<int>.Transpose(X, new int[] { 2, 0, 1 });
            Assert.Equal(4, tX.Dimensions[0]);
            Assert.Equal(3, tX.Dimensions[2]);
            Assert.Throws<ArgumentException>(() => Tensor<int>.Transpose(X, new int[] { 0, 0, 2 }));
            Assert.Throws<ArgumentException>(() => Tensor<int>.Transpose(X, new int[] { 0 }));
            Assert.Throws<ArgumentException>(() => Tensor<int>.Transpose(X, new int[] { 2, 6 }));
        }

        [Fact]
        public void CanConcat()
        {
            var x = DenseTensor<float>.OfValues(new float[2, 3] { { 0.6580f, -1.0969f, -0.4614f }, { -0.1034f, -0.5790f, 0.149f } });
            var y = Tensor<float>.Concat(x, x, 0);
            Assert.NotNull(y);
            y = Tensor<float>.Concat(x, x, 1);
            Assert.NotNull(y);
        }

        [Fact]
        public void CanUnsqueeze()
        {
            var X = (ITensor) DenseTensor<int>.Ones(3, 4, 5);
            var tX = X.Unsqueeze(new int[] { 1 });
            Assert.NotNull(tX);
;
        }
    }
}
