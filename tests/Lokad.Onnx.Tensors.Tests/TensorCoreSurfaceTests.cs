namespace Lokad.Onnx.Tensors.Tests;

using Lokad.Onnx.Tests.Support;

/// <summary>
/// Pins the public surface of the two ported tensor core files (Tensor.cs, DenseTensor.cs).
/// Any rewrite that drops, renames, or adds a public declaration fails here naming the
/// offender, so the hierarchy replacement cannot silently change the contract. Update the
/// embedded inventory explicitly when the surface intentionally changes.
/// </summary>
public class TensorCoreSurfaceTests
{
    internal static List<string> PortedCoreDeclarations(string root)
    {
        var decls = new List<string>();
        foreach (string name in new[] { "Tensor.cs", "DenseTensor.cs" })
        {
            string file = Path.Combine(root, "src", "Lokad.Onnx", name);
            foreach (string raw in File.ReadAllLines(file))
            {
                string line = string.Join(" ", raw.Split(new[] { (char)32, (char)9 }, StringSplitOptions.RemoveEmptyEntries)).Trim();
                if (!line.Contains("public ", StringComparison.Ordinal)) continue;
                if (line.StartsWith("///", StringComparison.Ordinal) || line.StartsWith("//", StringComparison.Ordinal)) continue;
                decls.Add(name + ":" + line);
            }
        }
        decls.Sort(StringComparer.Ordinal);
        return decls;
    }

    [Fact]
    public void PortedCoreSurface_MatchesInventory()
    {
        var actual = PortedCoreDeclarations(TestSupport.RepoRoot());
        var expected = new HashSet<string>(Expected, StringComparer.Ordinal);
        var missing = Expected.Where((string e) => !actual.Contains(e)).ToList();
        var added = actual.Where((string a) => !expected.Contains(a)).ToList();
        Assert.True(missing.Count == 0 && added.Count == 0,
            "Surface drift.\nMissing:\n" + string.Join("\n", missing.Take(20)) + "\nAdded:\n" + string.Join("\n", added.Take(20)));
    }

    static readonly string[] Expected = new string[]
    {
        @"DenseTensor.cs:public DenseTensor(int length) : base(length)",
        @"DenseTensor.cs:public DenseTensor(Memory<T> memory, ReadOnlySpan<int> dimensions, bool reverseStride)",
        @"DenseTensor.cs:public DenseTensor(Memory<T> memory, ReadOnlySpan<int> dimensions)",
        @"DenseTensor.cs:public DenseTensor(ReadOnlySpan<int> dimensions, bool reverseStride) : base(dimensions, reverseStride)",
        @"DenseTensor.cs:public DenseTensor(ReadOnlySpan<int> dimensions) : this(dimensions, false)",
        @"DenseTensor.cs:public Memory<T> Buffer => memory;",
        @"DenseTensor.cs:public override DenseTensor<T> ToDenseTensor()",
        @"DenseTensor.cs:public override T GetValue(int index)",
        @"DenseTensor.cs:public override Tensor<T> Clone()",
        @"DenseTensor.cs:public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions)",
        @"DenseTensor.cs:public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)",
        @"DenseTensor.cs:public override void SetValue(int index, T value)",
        @"DenseTensor.cs:public static DenseTensor<T> OfShape(params int[] dims) => new DenseTensor<T>((ReadOnlySpan<int>) dims);",
        @"DenseTensor.cs:public static DenseTensor<T> OfValues(Array data) => data.ToTensor<T>();",
        @"DenseTensor.cs:public static DenseTensor<T> OfValues(ReadOnlySpan<T> values, int[] dims)",
        @"DenseTensor.cs:public static DenseTensor<T> OfValues(T[] values)",
        @"DenseTensor.cs:public static DenseTensor<T> Scalar(T value) => new DenseTensor<T>(new T[1] { value }, Array.Empty<int>());",
        @"DenseTensor.cs:public unsafe class DenseTensor<T> : Tensor<T> where T : unmanaged",
        @"Tensor.cs:public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor",
        @"Tensor.cs:public abstract T GetValue(int index);",
        @"Tensor.cs:public abstract Tensor<T> Clone();",
        @"Tensor.cs:public abstract Tensor<T> Reshape(ReadOnlySpan<int> dimensions);",
        @"Tensor.cs:public abstract Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions) where TResult : unmanaged;",
        @"Tensor.cs:public abstract void SetValue(int index, T value);",
        @"Tensor.cs:public bool IsFixedSize => true;",
        @"Tensor.cs:public bool IsReadOnly => false;",
        @"Tensor.cs:public bool IsReversedStride => isReversedStride;",
        @"Tensor.cs:public bool IsString { get { return ElementType == TensorElementType.String; } }",
        @"Tensor.cs:public bool IsString { get; private set; }",
        @"Tensor.cs:public bool ShapeEquals(Tensor<T> t) => this.dimensions.SequenceEqual(t.dimensions);",
        @"Tensor.cs:public class TensorBase",
        @"Tensor.cs:public class TensorElementTypeInfo",
        @"Tensor.cs:public class TensorTypeInfo",
        @"Tensor.cs:public enum TensorElementType",
        @"Tensor.cs:public int GetStorageIndex(int[] indices) => this switch",
        @"Tensor.cs:public int Rank => dimensions.Length;",
        @"Tensor.cs:public int TypeSize { get; private set; }",
        @"Tensor.cs:public int TypeSize { get; private set; }",
        @"Tensor.cs:public int[] GetCoordinates(int offset)",
        @"Tensor.cs:public int[] SliceAxes(params SliceIndex[] input_slices)",
        @"Tensor.cs:public long Length => length;",
        @"Tensor.cs:public Memory<T> Storage => this switch",
        @"Tensor.cs:public ReadOnlySpan<int> Dimensions => dimensions;",
        @"Tensor.cs:public ReadOnlySpan<int> Strides => strides;",
        @"Tensor.cs:public SliceIndex[] ExpandEllipsis(SliceIndex[] slices)",
        @"Tensor.cs:public static Array CreateElementArray(TensorElementType elementType, int length) => elementType switch",
        @"Tensor.cs:public static bool Equals(Tensor<T> left, Tensor<T> right)",
        @"Tensor.cs:public static int Compare(Tensor<T> left, Tensor<T> right)",
        @"Tensor.cs:public static int ElementByteSize(TensorElementType elementType) => elementType switch",
        @"Tensor.cs:public static ITensor CreateDenseTensor(TensorElementType elementType, Array data, int[] dims) => elementType switch",
        @"Tensor.cs:public static Tensor<float> Arange(float start, float stop, float step)",
        @"Tensor.cs:public static Tensor<float> Arange(float start, float stop) => Arange(start, stop, 1.0f);",
        @"Tensor.cs:public static Tensor<float> Rand(params int[] dims)",
        @"Tensor.cs:public static Tensor<int> Arange(int start, int stop, int step)",
        @"Tensor.cs:public static Tensor<int> Arange(int start, int stop) => Arange(start, stop, 1);",
        @"Tensor.cs:public static Tensor<int> RandN(params int[] dims)",
        @"Tensor.cs:public static Tensor<T> Ones(params int[] dims)",
        @"Tensor.cs:public static Tensor<T> Zeros(params int[] dims)",
        @"Tensor.cs:public static TensorElementTypeInfo? GetElementTypeInfo(TensorElementType elementType)",
        @"Tensor.cs:public static TensorTypeInfo? GetTypeInfo(Type type)",
        @"Tensor.cs:public string Name { get; set; } = """";",
        @"Tensor.cs:public string PrintData(bool includeWhitespace)",
        @"Tensor.cs:public string PrintShape() => ""["" + string.Join(',', dimensions) + ""]"";",
        @"Tensor.cs:public T this[params Index[] indices]",
        @"Tensor.cs:public T this[params int[] indices]",
        @"Tensor.cs:public Tensor<T> PadLeft() => InsertDim(0);",
        @"Tensor.cs:public Tensor<T> PadRight() => InsertDim(Rank);",
        @"Tensor.cs:public Tensor<T> Reshape(params int[] dims) => Reshape((ReadOnlySpan<int>)dims);",
        @"Tensor.cs:public Tensor<T> this[params SliceIndex[] indices]",
        @"Tensor.cs:public Tensor<T> WithName(string name)",
        @"Tensor.cs:public TensorDimensionsIterator GetDimensionsIterator() => GetDimensionsIterator(..);",
        @"Tensor.cs:public TensorDimensionsIterator GetDimensionsIterator(Range r) => new TensorDimensionsIterator(dimensions[r]);",
        @"Tensor.cs:public TensorElementType ElementType { get; } = GetTypeInfo(typeof(T))?.ElementType ?? throw new NotSupportedException($""Tensor element type {typeof(T).Name} is not supported."");",
        @"Tensor.cs:public TensorElementType ElementType { get; private set; }",
        @"Tensor.cs:public TensorElementTypeInfo(Type type, int typeSize)",
        @"Tensor.cs:public TensorSlice<T> Slice(params SliceIndex[] indices) => new TensorSlice<T>(this, ExpandEllipsis(indices));",
        @"Tensor.cs:public TensorTypeInfo? GetTypeInfo()",
        @"Tensor.cs:public TensorTypeInfo(TensorElementType elementType, int typeSize)",
        @"Tensor.cs:public Type PrimitiveType { get; } = typeof(T);",
        @"Tensor.cs:public Type TensorType { get; private set; }",
        @"Tensor.cs:public virtual BroadcastedTensor<T> BroadcastDim(int dim, int size)",
        @"Tensor.cs:public virtual DenseTensor<T> ToDenseTensor()",
        @"Tensor.cs:public virtual T this[ReadOnlySpan<int> indices]",
        @"Tensor.cs:public virtual Tensor<T> CloneEmpty()",
        @"Tensor.cs:public virtual Tensor<T> CloneEmpty(ReadOnlySpan<int> dimensions)",
        @"Tensor.cs:public virtual Tensor<T> InsertDim(int dim)",
        @"Tensor.cs:public virtual Tensor<T> RemoveDim(int dim)",
        @"Tensor.cs:public virtual Tensor<TResult> CloneEmpty<TResult>() where TResult : unmanaged",
        @"Tensor.cs:public virtual void Fill(T value)",
    };
}
