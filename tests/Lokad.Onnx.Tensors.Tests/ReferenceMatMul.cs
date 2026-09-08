namespace Lokad.Onnx.Tensors.Tests;

using System;

// Naive managed 2D matrix-multiplication oracles, moved verbatim from the
// shipped core (Tensor<T>.MatMul2D_managed) so tests keep independent
// expected values without growing the distributed assembly.

static class ReferenceMatMul
{
    public static Tensor<int> Managed(Tensor<int> x, Tensor<int> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        int rA = x.Dimensions[0];
        int cA = x.Dimensions[1];
        int cB = y.Dimensions[1];
        var output = DenseTensor<int>.OfShape(new int[] { rA, cB });
        int temp;
       
        for (int i = 0; i < rA; i++)
        {
            for (int j = 0; j < cB; j++)
            {
                temp = 0;
                for (int k = 0; k < cA; k++)
                {
                    temp += x[i, k] * y[k, j];
                }
                output.SetValue((i * rA + j), temp);
            }
        }
        return output;
    }

    public static Tensor<float> Managed(Tensor<float> x, Tensor<float> y)
    {
        if (x.Rank != 2) throw new ArgumentException(nameof(x), "The rank of this tensor is not 2.");
        if (y.Rank != 2) throw new ArgumentException(nameof(y), "The rank of this tensor is not 2.");
        if (x.Dimensions[1] != y.Dimensions[0]) throw new ArgumentException("The number of columns in the first matrix is not equal to the number of rows in the second matrix.");
        int rA = x.Dimensions[0];
        int cA = x.Dimensions[1];
        int cB = y.Dimensions[1];
        var output = DenseTensor<float>.OfShape(new int[] { rA, cB });
        float temp;

        for (int i = 0; i < rA; i++)
        {
            for (int j = 0; j < cB; j++)
            {
                temp = 0;
                for (int k = 0; k < cA; k++)
                {
                    temp += x[i, k] * y[k, j];
                }
                output.SetValue((i * rA + j), temp);
            }
        }
        return output;
    }
}
