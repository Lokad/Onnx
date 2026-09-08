namespace Lokad.Onnx;

using System;
using System.Collections;

/// <summary>
/// Data-free tensor descriptor: retains a name, element type and shape for
/// graph input/output slots without allocating element storage. Only metadata
/// members work; every data access throws InvalidOperationException so a
/// descriptor can never be mistaken for produced values.
/// </summary>
public sealed class TensorDesc : ITensor
{
    public TensorDesc(string name, TensorElementType elementType, int[] dims)
    {
        Name = name;
        ElementType = elementType;
        this.dims = (int[])dims.Clone();
        checked
        {
            long n = 1;
            foreach (var d in dims) n *= d;
            Length = n;
        }
    }

    public TensorDesc(OnnxValueInfo info) : this(info.Name, info.ElementType, info.Dims)
    {
    }

    public string Name { get; set; }

    public TensorElementType ElementType { get; }

    public Type PrimitiveType => ElementType switch
    {
        TensorElementType.Bool => typeof(bool),
        TensorElementType.Int8 => typeof(sbyte),
        TensorElementType.UInt8 => typeof(byte),
        TensorElementType.Int16 => typeof(short),
        TensorElementType.UInt16 => typeof(ushort),
        TensorElementType.Int32 => typeof(int),
        TensorElementType.UInt32 => typeof(uint),
        TensorElementType.Int64 => typeof(long),
        TensorElementType.UInt64 => typeof(ulong),
        TensorElementType.Float => typeof(float),
        TensorElementType.Double => typeof(double),
        TensorElementType.Float16 => typeof(Float16),
        TensorElementType.BFloat16 => typeof(BFloat16),
        TensorElementType.Complex64 => typeof(System.Numerics.Complex),
        TensorElementType.String => typeof(string),
        _ => throw new NotSupportedException($"No primitive type for element type {ElementType}."),
    };

    /// <summary>A copy of the shape metadata; mutating it affects nothing.</summary>
    public int[] Dims => (int[])dims;

    private readonly int[] dims;

    public int Rank => dims.Length;

    public long Length { get; }

    public string PrintShape() => "[" + string.Join(",", Dims) + "]";

    static InvalidOperationException NoData(string member) =>
        new InvalidOperationException($"Tensor descriptor {member} has no data; it only describes shape metadata.");

    public ITensor Clone() => throw NoData(nameof(Clone));

    public ITensor CloneEmpty() => throw NoData(nameof(CloneEmpty));

    public ITensor CloneEmpty<U>() where U : unmanaged => throw NoData(nameof(CloneEmpty));

    public ITensor Reshape(params int[] shape) => throw NoData(nameof(Reshape));

    public ITensor Slice(string indices) => throw NoData(nameof(Slice));

    public ITensor InsertDim(int dim) => throw NoData(nameof(InsertDim));

    public ITensor RemoveDim(int dim) => throw NoData(nameof(RemoveDim));

    public ITensor BroadcastDim(int dim, int size) => throw NoData(nameof(BroadcastDim));

    public ITensor ToDenseTensor() => throw NoData(nameof(ToDenseTensor));

    public Array ToArray() => throw NoData(nameof(ToArray));

    public object this[params int[] indices]
    {
        get => throw NoData("indexer");
        set => throw NoData("indexer");
    }

    public ITensor this[params object[] indices]
    {
        get => throw NoData("indexer");
        set => throw NoData("indexer");
    }

    public object GetValue(int index) => throw NoData(nameof(GetValue));

    public void SetValue(int index, object? value) => throw NoData(nameof(SetValue));

    public string PrintData(bool includeWhitespace) => throw NoData(nameof(PrintData));

    IEnumerator IEnumerable.GetEnumerator() => throw NoData("enumeration");
}
