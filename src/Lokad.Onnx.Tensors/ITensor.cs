using Lokad.Onnx;
using System;
using System.Collections;
using System.Linq;
using System.Text;

namespace Lokad.Onnx
{
    public interface ITensor : IEnumerable
    {
        string Name { get; set; }

        TensorElementType ElementType { get; }
        
        Type PrimitiveType {get; }
        
        int[] Dims { get; }

        int Rank { get; }

        ITensor Clone();

        ITensor Reshape(params int[] shape);

        ITensor Slice(string indices);

        ITensor InsertDim(int dim);

        ITensor RemoveDim(int dim);

        ITensor BroadcastDim(int dim, int size);

        ITensor ToDenseTensor();

        ITensor PadLeft() => InsertDim(0);
        
        object this[params int[] indices]
        {
            get;
            set;
        }

        ITensor this[params object[] indices]
        {
            get;
            set;
        }

        static void ThrowIfDifferentElementTypes(params ITensor[] tensors) => Array.ForEach(tensors, tensor =>
        {
            if (tensor.ElementType != tensors[0].ElementType) throw new ArgumentException("All tensors must have the same element type.");
        });
        
        static ITensor[] Broadcast(ITensor inA, ITensor inB)
        {
            var broadcastRank = Math.Max(inA.Rank, inB.Rank);
            var outA = inA.Clone();
            var outB = inB.Clone();
            for (var i = 0; i < broadcastRank; i++)
            {
                var idxA = i - broadcastRank + inA.Rank;
                var idxB = i - broadcastRank + inB.Rank;
                if (i < broadcastRank - inA.Rank)
                {
                    outA = outA.PadLeft();
                    outA = outA.BroadcastDim(0, inB.Dims[idxB]);
                }
                else if (i < broadcastRank - inB.Rank)
                {
                    outB = outB.PadLeft();
                    outB = outB.BroadcastDim(0, inA.Dims[idxA]);
                }
                else if (inA.Dims[idxA] == inB.Dims[idxB])
                {
                }
                else if (inA.Dims[idxA] == 1)
                {
                    outA = outA.BroadcastDim(i, inB.Dims[idxB]);
                }
                else if (inB.Dims[idxB] == 1)
                {
                    outB = outB.BroadcastDim(i, inA.Dims[idxA]);
                }
                else
                {
                    return Array.Empty<ITensor>();
                    //return OpResult.Failure(OpType.Broadcast, $"Trying to broadcast incompatible shapes: {inA.Dimensions.ToArray()} and {inB.Dimensions.ToArray()}");
                }
            }
            return new[] { outA, outB };
        }

        static bool Broadcast(ITensor x, ITensor y, out ITensor bx, out ITensor by)
        {
            var b = Broadcast(x, y);
            if (b.Length == 0)
            {
                bx = x;
                by = y;
                return false;
            }
            else
            {
                bx = b[0];
                by = b[1];  
                return true;
            }
        }
        string PrintShape() => "[" + string.Join(',', Dims) + "]";

        int Length => Dims.Aggregate((p, n) => p * n);

        string PrintData(bool includeWhitespace = true)
        {
            var builder = new StringBuilder();

            var strides = ArrayUtilities.GetStrides(Dims);
            var indices = new int[Rank];
            var innerDimension = Rank - 1;
            var innerLength = Dims[innerDimension];
            var outerLength = Length / innerLength;

            int indent = 0;
            for (int outerIndex = 0; outerIndex < Length; outerIndex += innerLength)
            {
                ArrayUtilities.GetIndices(strides, false, outerIndex, indices);

                while ((indent < innerDimension) && (indices[indent] == 0))
                {
                    // start up
                    if (includeWhitespace)
                    {
                        Indent(builder, indent);
                    }
                    indent++;
                    builder.Append('[');
                    if (includeWhitespace)
                    {
                        builder.AppendLine();
                    }
                }

                for (int innerIndex = 0; innerIndex < innerLength; innerIndex++)
                {
                    indices[innerDimension] = innerIndex;

                    if ((innerIndex == 0))
                    {
                        if (includeWhitespace)
                        {
                            Indent(builder, indent);
                        }
                        builder.Append('[');
                    }
                    else
                    {
                        builder.Append(',');
                    }
                    builder.Append(this[indices]);
                }
                builder.Append(']');

                for (int i = Rank - 2; i >= 0; i--)
                {
                    var lastIndex = Dims[i] - 1;
                    if (indices[i] == lastIndex)
                    {
                        // close out
                        --indent;
                        if (includeWhitespace)
                        {
                            builder.AppendLine();
                            Indent(builder, indent);
                        }
                        builder.Append(']');
                    }
                    else
                    {
                        builder.Append(',');
                        if (includeWhitespace)
                        {
                            builder.AppendLine();
                        }
                        break;
                    }
                }
            }
            return builder.ToString();
            void Indent(StringBuilder builder, int tabs, int spacesPerTab = 4)
            {
                for (int tab = 0; tab < tabs; tab++)
                {
                    for (int space = 0; space < spacesPerTab; space++)
                    {
                        builder.Append(' ');
                    }
                }
            }
        }

        string TensorNameDesc() => $"{Name}:{ElementType.ToString().ToLower()}:{string.Join("x",Dims.Select(d => d.ToString()))}";
    }
}

public class TensorInputShapeException : ArgumentException
{
    public ITensor Input { get; set; }
    public int[] Shape { get; set; }

    public string Name { get; set; }
    public TensorInputShapeException(string paramName, int[] shape, ITensor input) : base(paramName, $"The input parameter {paramName} has shape {input.PrintShape()} but is required to be {shape.Print()}.")
    {
        Name = paramName;
        Shape = shape;
        Input = input;
    }
}
