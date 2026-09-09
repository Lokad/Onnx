using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.Versioning;
using System.Numerics;
using System.Text;
using System.Drawing;

namespace Lokad.Onnx
{
    #region Enums
    /// <summary>
    /// Supported Tensor DataType
    /// </summary>
    public enum TensorElementType
    {
        Float = 1,
        UInt8 = 2,
        Int8 = 3,
        UInt16 = 4,
        Int16 = 5,
        Int32 = 6,
        Int64 = 7,
        String = 8,
        Bool = 9,
        Float16 = 10,
        Double = 11,
        UInt32 = 12,
        UInt64 = 13,
        Complex64 = 14,
        Complex128 = 15,
        BFloat16 = 16,
        DataTypeMax = 17,
        Sequence = 100
    }
    #endregion

    #region Types
    /// <summary>
    /// Maps a .NET element type to its tensor element kind and storage width.
    /// </summary>
    public class TensorTypeInfo
    {
        /// <summary>
        /// The tensor element kind.
        /// </summary>
        /// <value>The tensor element kind.</value>
        public TensorElementType ElementType { get; private set; }
        /// <summary>
        /// Storage width of one element in bytes
        /// </summary>
        /// <value>Storage width in bytes.</value>
        public int TypeSize { get; private set; }
        /// <summary>
        /// Whether the element type is string
        /// </summary>
        /// <value>True for the string element type.</value>
        public bool IsString { get { return ElementType == TensorElementType.String; } }
        /// <summary>
        /// Creates a type trait record.
        /// </summary>
        /// <param name="elementType">Tensor element kind.</param>
        /// <param name="typeSize">Storage width in bytes.</param>
        public TensorTypeInfo(TensorElementType elementType, int typeSize)
        {
            ElementType = elementType;
            TypeSize = typeSize;
        }
    }

    /// <summary>
    /// Maps a tensor element kind back to its .NET type and storage width.
    /// </summary>
    public class TensorElementTypeInfo
    {
        /// <summary>
        /// The .NET element type.
        /// </summary>
        /// <value>The .NET element type.</value>
        public Type TensorType { get; private set; }
        /// <summary>
        /// Storage width of one element in bytes
        /// </summary>
        /// <value>Storage width in bytes.</value>
        public int TypeSize { get; private set; }
        /// <summary>
        /// Whether the element type is string
        /// </summary>
        /// <value>True for the string element type.</value>
        public bool IsString { get; private set; }
        /// <summary>
        /// Creates a type trait record.
        /// </summary>
        /// <param name="type">.NET element type.</param>
        /// <param name="typeSize">Storage width in bytes.</param>
        public TensorElementTypeInfo(Type type, int typeSize)
        {
            TensorType = type;
            TypeSize = typeSize;
            IsString = type == typeof(string);
        }
    }

    /// <summary>
    /// Base of every tensor value. Hosts the element-type trait tables shared by import, export, and dispatch.
    /// </summary>
    public class TensorBase
    {
        /// <summary>
        /// Exact backing array when this value densely owns precisely its logical
        /// contents, else null. Views and slices never qualify,
        /// so returning their storage through this probe is impossible.
        /// </summary>
        internal virtual Array? OwnedBufferArray() => null;

        private static readonly Dictionary<Type, TensorTypeInfo> typeInfoMap;

        private static readonly Dictionary<TensorElementType, TensorElementTypeInfo> tensorElementTypeInfoMap;

        static TensorBase()
        {
            typeInfoMap = new Dictionary<Type, TensorTypeInfo>()
            {
                { typeof(float), new TensorTypeInfo( TensorElementType.Float, sizeof(float)) },
                { typeof(byte), new TensorTypeInfo( TensorElementType.UInt8, sizeof(byte)) },
                { typeof(sbyte), new TensorTypeInfo( TensorElementType.Int8, sizeof(sbyte)) },
                { typeof(ushort), new TensorTypeInfo( TensorElementType.UInt16, sizeof(ushort)) },
                { typeof(short), new TensorTypeInfo( TensorElementType.Int16, sizeof(short)) },
                { typeof(int), new TensorTypeInfo( TensorElementType.Int32, sizeof(int)) },
                { typeof(long), new TensorTypeInfo( TensorElementType.Int64, sizeof(long)) },
                { typeof(string), new TensorTypeInfo( TensorElementType.String, -1) },
                { typeof(bool), new TensorTypeInfo( TensorElementType.Bool, sizeof(bool)) },
                { typeof(Half), new TensorTypeInfo( TensorElementType.Float16, sizeof(ushort)) },
                { typeof(double), new TensorTypeInfo( TensorElementType.Double, sizeof(double)) },
                { typeof(uint), new TensorTypeInfo( TensorElementType.UInt32, sizeof(uint)) },
                { typeof(ulong), new TensorTypeInfo( TensorElementType.UInt64, sizeof(ulong)) },
                { typeof(BFloat16), new TensorTypeInfo( TensorElementType.BFloat16, sizeof(ushort)) },
                { typeof(System.Numerics.Complex), new TensorTypeInfo( TensorElementType.Complex64, sizeof(double) * 2) }
            };

            tensorElementTypeInfoMap = new Dictionary<TensorElementType, TensorElementTypeInfo>();
            foreach (var info in typeInfoMap)
            {
                tensorElementTypeInfoMap.Add(info.Value.ElementType, new TensorElementTypeInfo(info.Key, info.Value.TypeSize));
            }
        }

        private readonly Type _primitiveType;
        /// <summary>
        /// Records the element type governed by this tensor
        /// </summary>
        /// <param name="primitiveType">primitive type the deriving class is using</param>
        protected TensorBase(Type primitiveType)
        {
            // Should hold as we rely on this to pass arrays of these
            // types to native code
            unsafe
            {
                Debug.Assert(sizeof(ushort) == sizeof(Half));
                Debug.Assert(sizeof(ushort) == sizeof(BFloat16));
            }
            _primitiveType = primitiveType;
        }

        /// <summary>
        /// Looks up the trait record for a .NET element type
        /// </summary>
        /// <param name="type"></param>
        /// <returns>TensorTypeInfo or null if not supported</returns>
        public static TensorTypeInfo? GetTypeInfo(Type type)
        {
            TensorTypeInfo? result = null;
            typeInfoMap.TryGetValue(type, out result);
            return result;
        }

        /// <summary>
        /// Looks up the trait record for a tensor element kind
        /// </summary>
        /// <param name="elementType">type enum</param>
        /// <returns>instance of TensorElementTypeInfo or null if not found</returns>
        public static TensorElementTypeInfo? GetElementTypeInfo(TensorElementType elementType)
        {
            TensorElementTypeInfo? result = null;
            tensorElementTypeInfoMap.TryGetValue(elementType, out result);
            return result;
        }

        /// <summary>
        /// Fixed storage size in bytes for element types with one, or -1 for
        /// variable-size (String) and unsupported types. Single home for the
        /// byte-size table used by import validation and external-data reads.
        /// </summary>
        public static int ElementByteSize(TensorElementType elementType) => elementType switch
        {
            TensorElementType.Bool => 1,
            TensorElementType.Int8 => 1,
            TensorElementType.UInt8 => 1,
            TensorElementType.Int16 => 2,
            TensorElementType.UInt16 => 2,
            TensorElementType.Float16 => 2,
            TensorElementType.BFloat16 => 2,
            TensorElementType.Int32 => 4,
            TensorElementType.UInt32 => 4,
            TensorElementType.Float => 4,
            TensorElementType.Int64 => 8,
            TensorElementType.UInt64 => 8,
            TensorElementType.Double => 8,
            TensorElementType.Complex64 => 8,
            TensorElementType.Complex128 => 16,
            _ => -1,
        };

        /// <summary>
        /// Allocates an empty dense element array for the element type.
        /// Throws NotSupportedException for types with no dense representation.
        /// </summary>
        public static Array CreateElementArray(TensorElementType elementType, int length) => elementType switch
        {
            TensorElementType.Bool => new bool[length],
            TensorElementType.Int8 => new sbyte[length],
            TensorElementType.UInt8 => new byte[length],
            TensorElementType.Int16 => new short[length],
            TensorElementType.UInt16 => new ushort[length],
            TensorElementType.Float16 => new Half[length],
            TensorElementType.BFloat16 => new BFloat16[length],
            TensorElementType.Int32 => new int[length],
            TensorElementType.UInt32 => new uint[length],
            TensorElementType.Float => new float[length],
            TensorElementType.Int64 => new long[length],
            TensorElementType.UInt64 => new ulong[length],
            TensorElementType.Double => new double[length],
            _ => throw new NotSupportedException($"Unsupported element type {elementType}; no dense array representation."),
        };

        /// <summary>
        /// Materializes a dense tensor of the element type over caller-owned
        /// data. Throws ArgumentException for types with no dense tensor form.
        /// </summary>
        public static ITensor CreateDenseTensor(TensorElementType elementType, Array data, int[] dims) => elementType switch
        {
            TensorElementType.Bool => new DenseTensor<bool>(memory: (bool[])data, dims),
            TensorElementType.Int8 => new DenseTensor<sbyte>(memory: (sbyte[])data, dims),
            TensorElementType.UInt8 => new DenseTensor<byte>(memory: (byte[])data, dims),
            TensorElementType.Int16 => new DenseTensor<short>(memory: (short[])data, dims),
            TensorElementType.UInt16 => new DenseTensor<ushort>(memory: (ushort[])data, dims),
            TensorElementType.Int32 => new DenseTensor<int>(memory: (int[])data, dims),
            TensorElementType.UInt32 => new DenseTensor<uint>(memory: (uint[])data, dims),
            TensorElementType.Int64 => new DenseTensor<long>(memory: (long[])data, dims),
            TensorElementType.UInt64 => new DenseTensor<ulong>(memory: (ulong[])data, dims),
            TensorElementType.Float => new DenseTensor<float>(memory: (float[])data, dims),
            TensorElementType.Double => new DenseTensor<double>(memory: (double[])data, dims),
            TensorElementType.Float16 => new DenseTensor<Half>(memory: (Half[])data, dims),
            TensorElementType.BFloat16 => new DenseTensor<BFloat16>(memory: (BFloat16[])data, dims),
            TensorElementType.Complex64 => new DenseTensor<Complex>(memory: (Complex[])data, dims),
            _ => throw new ArgumentException($"Cannot convert model tensor of element type {elementType}."),
        };

        /// <summary>
        /// Returns the trait record for this tensor's element type
        /// </summary>
        /// <returns></returns>
        public TensorTypeInfo? GetTypeInfo()
        {
            return GetTypeInfo(_primitiveType);
        }
    }

    #endregion

    /// <summary>
    /// Represents a multi-dimensional collection of objects of type T that can be accessed by indices.
    /// </summary>
    /// <typeparam name="T">type contained within the Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
    [DebuggerDisplay("{PrintShape()}")]
    public abstract partial class Tensor<T> : TensorBase, IList, IList<T>, IReadOnlyList<T>, IStructuralComparable, IStructuralEquatable, ITensor, INumericTensor
    where T : unmanaged
    {
        internal static T Zero
        {
            get
            {
                if (typeof(T) == typeof(bool))
                {
                    return (T)(object)(false);
                }
                else if (typeof(T) == typeof(byte))
                {
                    return (T)(object)(byte)(0);
                }
                else if (typeof(T) == typeof(char))
                {
                    return (T)(object)(char)(0);
                }
                else if (typeof(T) == typeof(decimal))
                {
                    return (T)(object)(decimal)(0);
                }
                else if (typeof(T) == typeof(double))
                {
                    return (T)(object)(double)(0);
                }
                else if (typeof(T) == typeof(float))
                {
                    return (T)(object)(float)(0);
                }
                else if (typeof(T) == typeof(int))
                {
                    return (T)(object)(int)(0);
                }
                else if (typeof(T) == typeof(long))
                {
                    return (T)(object)(long)(0);
                }
                else if (typeof(T) == typeof(sbyte))
                {
                    return (T)(object)(sbyte)(0);
                }
                else if (typeof(T) == typeof(short))
                {
                    return (T)(object)(short)(0);
                }
                else if (typeof(T) == typeof(uint))
                {
                    return (T)(object)(uint)(0);
                }
                else if (typeof(T) == typeof(ulong))
                {
                    return (T)(object)(ulong)(0);
                }
                else if (typeof(T) == typeof(ushort))
                {
                    return (T)(object)(ushort)(0);
                }
                else if (typeof(T) == typeof(Half))
                {
                    return (T)(object)Half.Zero;
                }
                else if (typeof(T) == typeof(BFloat16))
                {
                    return (T)(object)BFloat16.Zero;
                }
                else if (typeof(T) == typeof(System.Numerics.Complex))
                {
                    return (T)(object)System.Numerics.Complex.Zero;
                }
                else if (typeof(T) == typeof(string))
                {
                    return (T)(object)("0");
                }
                throw new NotSupportedException();
            }
        }

        internal static T One
        {
            get
            {
                if (typeof(T) == typeof(bool))
                {
                    return (T)(object)(true);
                }
                else if (typeof(T) == typeof(byte))
                {
                    return (T)(object)(byte)(1);
                }
                else if (typeof(T) == typeof(char))
                {
                    return (T)(object)(char)(1);
                }
                else if (typeof(T) == typeof(decimal))
                {
                    return (T)(object)(decimal)(1);
                }
                else if (typeof(T) == typeof(double))
                {
                    return (T)(object)(double)(1);
                }
                else if (typeof(T) == typeof(float))
                {
                    return (T)(object)(float)(1);
                }
                else if (typeof(T) == typeof(int))
                {
                    return (T)(object)(int)(1);
                }
                else if (typeof(T) == typeof(long))
                {
                    return (T)(object)(long)(1);
                }
                else if (typeof(T) == typeof(sbyte))
                {
                    return (T)(object)(sbyte)(1);
                }
                else if (typeof(T) == typeof(short))
                {
                    return (T)(object)(short)(1);
                }
                else if (typeof(T) == typeof(uint))
                {
                    return (T)(object)(uint)(1);
                }
                else if (typeof(T) == typeof(ulong))
                {
                    return (T)(object)(ulong)(1);
                }
                else if (typeof(T) == typeof(ushort))
                {
                    return (T)(object)(ushort)(1);
                }
                else if (typeof(T) == typeof(Half))
                {
                    return (T)(object)Half.One;
                }
                else if (typeof(T) == typeof(BFloat16))
                {
                    return (T)(object)BFloat16.One;
                }
                else if (typeof(T) == typeof(System.Numerics.Complex))
                {
                    return (T)(object)System.Numerics.Complex.One;
                }
                else if (typeof(T) == typeof(string))
                {
                    return (T)(object)("1");
                }

                throw new NotSupportedException();
            }
        }

        internal readonly int[] dimensions;
        internal readonly int[] strides;
        private readonly bool isReversedStride;

        private readonly long length;

        /// <summary>
        /// Creates a rank-1 tensor of the specified length
        /// </summary>
        /// <param name="length">Size of the 1-dimensional tensor</param>
        protected Tensor(int length) : base(typeof(T))
        {
            dimensions = new[] { length };
            strides = new[] { 1 };
            isReversedStride = false;
            this.length = length;
        }

        /// <summary>
        /// Creates an n-dimensional tensor with the specified dimensions and layout.
        /// ReverseStride=true gives a stride of 1-element width to the first dimension (0).  
        /// ReverseStride=false gives a stride of 1-element width to the last dimension (n-1).
        /// </summary>
        /// <param name="dimensions">
        /// An span of integers that represent the size of each dimension of the Tensor to create.</param>
        /// <param name="reverseStride">
        /// False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension 
        /// is most minor (closest together): akin to row-major in a rank-2 tensor.  
        /// True to indicate that the last dimension is most major (farthest apart) and the first dimension is most 
        /// minor (closest together): akin to column-major in a rank-2 tensor.</param>
        /// <remarks>If you pass `null` for dimensions it will implicitly convert to an empty ReadOnlySpan, which is 
        /// equivalent to the dimensions for a scalar value.</remarks>
        protected Tensor(ReadOnlySpan<int> dimensions, bool reverseStride) : base(typeof(T))
        {
            this.dimensions = new int[dimensions.Length];
            checked
            {
                long size = 1;
                for (int i = 0; i < dimensions.Length; i++)
                {
                    if (dimensions[i] < 0)
                    {
                        throw new ArgumentOutOfRangeException(nameof(dimensions), "Dimensions must be non-negative");
                    }
                    this.dimensions[i] = dimensions[i];
                    size *= dimensions[i];
                }

                this.strides = ArrayUtilities.GetStrides(dimensions, reverseStride);
                isReversedStride = reverseStride;

                if (size > int.MaxValue) throw new ArgumentException("Tensor element count exceeds maximum backing-store length.", nameof(dimensions));
                length = size;
            }
        }

        /// <summary>
        /// Creates a tensor shaped like an array; the array contents are ignored.  
        /// ReverseStride=true gives a stride of 1-element width to the first dimension (0).  
        /// ReverseStride=false gives a stride of 1-element width to the last dimension (n-1).
        /// </summary>
        /// <param name="fromArray">Array from which to derive dimensions.</param>
        /// <param name="reverseStride">
        /// False (default) to indicate that the first dimension is most major (farthest apart) and the 
        /// last dimension is most minor (closest together): akin to row-major in a rank-2 tensor.  
        /// True to indicate that the last dimension is most major (farthest apart) and the first dimension 
        /// is most minor (closest together): akin to column-major in a rank-2 tensor.</param>
        protected Tensor(Array fromArray, bool reverseStride) : base(typeof(T))
        {
            if (fromArray == null)
            {
                throw new ArgumentNullException(nameof(fromArray));
            }

            dimensions = new int[fromArray.Rank];
            checked
            {
                long size = 1;
                for (int i = 0; i < dimensions.Length; i++)
                {
                    dimensions[i] = fromArray.GetLength(i);
                    size *= dimensions[i];
                }

                strides = ArrayUtilities.GetStrides(dimensions, reverseStride);
                isReversedStride = reverseStride;

                if (size > int.MaxValue) throw new ArgumentException("Tensor element count exceeds maximum backing-store length.", nameof(fromArray));
                length = size;
            }
        }

        /// <summary>
        /// Total length of the Tensor.
        /// </summary>
        public long Length => length;

        /// <summary>
        /// Rank of the tensor: number of dimensions.
        /// </summary>
        public int Rank => dimensions.Length;

        /// <summary>
        /// True if strides are reversed (AKA Column-major)
        /// </summary>
        public bool IsReversedStride => isReversedStride;

        /// <summary>
        /// Returns a readonly view of the dimensions of this tensor.
        /// </summary>
        public ReadOnlySpan<int> Dimensions => dimensions;

        /// <summary>
        /// Returns a readonly view of the strides of this tensor.
        /// </summary>
        public ReadOnlySpan<int> Strides => strides;

        /// <summary>
        /// Sets all elements in Tensor to <paramref name="value"/>.
        /// </summary>
        /// <param name="value">Value to fill</param>
        public virtual void Fill(T value)
        {
            for (int i = 0; i < Length; i++)
            {
                SetValue(i, value);
            }
        }

        #region Cloning

        /// <summary>
        /// Duplicates every element into new backing storage.
        /// </summary>
        /// <returns>A copy holding the same values.</returns>
        public abstract Tensor<T> Clone();

        /// <summary>
        /// Creates a new Tensor with the same layout and dimensions as this tensor with elements initialized to their default value.
        /// </summary>
        /// <returns>A new Tensor with the same layout and dimensions as this tensor with elements initialized to their default value.</returns>
        public virtual Tensor<T> CloneEmpty()
        {
            return CloneEmpty<T>(dimensions);
        }

        /// <summary>
        /// Creates a new Tensor with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new Tensor with the same layout as this tensor and specified <paramref name="dimensions"/> with elements initialized to their default value.</returns>
        public virtual Tensor<T> CloneEmpty(ReadOnlySpan<int> dimensions)
        {
            return CloneEmpty<T>(dimensions);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the same layout and size as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained within the new Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
        /// <returns>A new Tensor with the same layout and dimensions as this tensor with elements of <typeparamref name="TResult"/> type initialized to their default value.</returns>
        public virtual Tensor<TResult> CloneEmpty<TResult>() where TResult : unmanaged
        {
            return CloneEmpty<TResult>(dimensions);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained within the new Tensor.  Typically a value type such as int, double, float, etc.</typeparam>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new Tensor with the same layout as this tensor of specified <paramref name="dimensions"/> with elements of <typeparamref name="TResult"/> type initialized to their default value.</returns>
        public abstract Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions) where TResult : unmanaged;
        
        #endregion

        /// <summary>
        /// Reinterprets this tensor under new dimensions, sharing storage when possible.
        /// </summary>
        /// <param name="dimensions">An span of integers that represent the size of each dimension of the Tensor to create.</param>
        /// <returns>A new tensor that reinterprets this tensor with different dimensions.</returns>
        public abstract Tensor<T> Reshape(ReadOnlySpan<int> dimensions);

        /// <summary>
        /// Reads the element at the specified indices
        /// </summary>
        /// <param name="indices">A one-dimensional array of integers that represent the indices specifying the position of the element to get.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>

        #region Indexing

        /// <summary>
        /// Reads the element at a flat index (the stride dot product of coordinates).
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        public abstract T GetValue(int index);

        /// <summary>
        /// Writes the element at a flat index (the stride dot product of coordinates).
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <param name="value">The new value to set at the specified position in this Tensor.</param>
        public abstract void SetValue(int index, T value);

        public T this[params int[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            get => this[(ReadOnlySpan<int>)indices];
            
            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            set => this[(ReadOnlySpan<int>)indices] = value;
        }

        /// <summary>
        /// Reads the element at the specified indices
        /// </summary>
        /// <param name="indices">A span integers that represent the indices specifying the position of the element to get.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        
        public virtual T this[ReadOnlySpan<int> indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            get => GetValue(ArrayUtilities.GetIndex(strides, indices));
            
            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            set => SetValue(ArrayUtilities.GetIndex(strides, indices), value);
        }

        public T this[params Index[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            get
            {
                return this[ResolveIndexes(indices)];
            }

            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            set
            {
                this[ResolveIndexes(indices)] = value;
            }
        }

        /// <summary>
        /// Converts System Index coordinates (including from-end ^ syntax) into plain
        /// positions, with the same bounds checks the inline query used to perform.
        /// </summary>
        private int[] ResolveIndexes(Index[] indices)
        {
            var resolved = new int[indices.Length];
            for (int n = 0; n < indices.Length; n++)
            {
                Index i = indices[n];
                if (i.Equals(^0)) resolved[n] = dimensions[n] - 1;
                else if ((i.Value >= dimensions[n]) || (i.IsFromEnd && (dimensions[n] - i.Value >= dimensions[n]))) throw new ArgumentException(n.ToString());
                else if (i.IsFromEnd) resolved[n] = dimensions[n] - i.Value;
                else resolved[n] = i.Value;
            }
            return resolved;
        }

        public Tensor<T> this[params SliceIndex[] indices]
        {
            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            get
            {
                return new TensorSlice<T>(this, indices);
            }
            
            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            set
            {
                var expanded = ExpandEllipsis(indices);
                var defs = new SliceDef[expanded.Length];
                for (var i = 0; i < expanded.Length; i++)
                {
                    defs[i] = expanded[i].ToSliceDef(dimensions[i]);
                }
                var sliceDims = SliceAxes(expanded);

                // A source sharing this backing store is snapshotted first so the
                // copy observes the original values rather than partially written ones.
                var src = SharesStorage(this, value) ? Snapshot(value) : value;
                foreach (var index in src.GetDimensionsIterator())
                {
                    this.SetValue(GetOffsetUnsafe(strides, sliceDims, defs, index), src[index]);
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        protected int GetOffsetUnsafe(int[] orig_strides, int[] slice_dims, SliceDef[] slices, ReadOnlySpan<int> indices)
        {
            int offset;
            // Scratch coordinates live on the stack: capacity is Rank, validated below.
            if (indices.Length > Rank) throw new ArgumentOutOfRangeException(nameof(indices), $"Too many coordinates for tensor rank {Rank}.");
            Span<int> coords = stackalloc int[Rank];
            int coordCount = indices.Length;
            indices.CopyTo(coords);
            var orig_ndim = orig_strides.Length;
            if (orig_ndim > slice_dims.Length && orig_ndim > indices.Length)
            {
                // Reduced dimensions are spliced back into the coordinates
                for (int i = 0; i < Rank; i++)
                {
                    var slice = slices[i];
                    if (slice.IsIndex)
                    {
                        if (coordCount >= Rank) throw new ArgumentOutOfRangeException(nameof(indices), "Too many coordinates for tensor rank {Rank}.");
                        for (int j = coordCount; j > i; j--) coords[j] = coords[j - 1];
                        coords[i] = 0;
                        coordCount++;
                    }
                }
            }
            offset = 0;
            unchecked
            {
                for (int i = 0; i < coordCount; i++)
                {
                    // Bounds were checked by the callers above, so this accumulation stays unchecked.
                    if (slices.Length <= i)
                    {
                        offset += orig_strides[i] * coords[i];
                        continue;
                    }

                    var slice = slices[i];
                    var start = slice.Start;
                    if (slice.IsIndex)
                        offset += orig_strides[i] * start; // reduced dimensions ignore the coordinate
                    else
                        offset += orig_strides[i] * (start + coords[i] * slice.Step);
                }
            }
            return offset;
        }
        #endregion

        #region ITensor support
        /// <summary>
        /// Returns this tensor with an added size-1 dimension at <paramref name="dim"/>.
        /// No data is copied: the result is a reshaped view that may share backing
        /// storage, so writes through either tensor may be visible in the other.
        /// </summary>
        /// <param name="dim">Insertion position from 0 through Rank inclusive.</param>
        /// <returns>A tensor with rank one higher and identical elements.</returns>
        /// <exception cref="IndexOutOfRangeException"><paramref name="dim"/> is negative or greater than Rank.</exception>
        public virtual Tensor<T> InsertDim(int dim)
        {
            if (dim < 0 || dim > Rank) throw new IndexOutOfRangeException(nameof(dim));
            var dims = dimensions.ToList();
            dims.Insert(dim, 1);
            return Reshape(dims.ToArray());
        }

        /// <summary>
        /// Returns this tensor with the size-1 dimension <paramref name="dim"/> removed.
        /// No data is copied: the result is a reshaped view that may share backing storage.
        /// </summary>
        /// <param name="dim">Index of a size-1 dimension.</param>
        /// <returns>A tensor with rank one lower and identical elements.</returns>
        /// <exception cref="IndexOutOfRangeException"><paramref name="dim"/> is outside the rank.</exception>
        /// <exception cref="ArgumentException">The dimension at <paramref name="dim"/> is not size 1.</exception>
        public virtual Tensor<T> RemoveDim(int dim)
        {
            if (dim >= Rank) throw new IndexOutOfRangeException(nameof(dim));
            if (dimensions[dim] != 1) throw new ArgumentException($"Dimension {dim} is size {dimensions[dim]}, not 1.");
            var dims = dimensions.ToList();
            dims.RemoveAt(dim);
            return Reshape(dims.ToArray());
        }

        /// <summary>
        /// Returns a broadcast view that repeats the size-1 dimension <paramref name="dim"/>
        /// <paramref name="size"/> times. No data is copied: the view shares the source
        /// storage, so a write through any broadcast position lands on the shared element.
        /// </summary>
        /// <param name="dim">Index of a size-1 dimension.</param>
        /// <param name="size">Replacement extent for that dimension.</param>
        /// <returns>A view with the same rank whose dimension <paramref name="dim"/> is <paramref name="size"/>.</returns>
        /// <exception cref="ArgumentException"><paramref name="dim"/> is outside the rank or does not index a size-1 dimension.</exception>
        public virtual BroadcastedTensor<T> BroadcastDim(int dim, int size)
        {
            if (dim >= Rank)
            {
                throw new ArgumentException($"The specified dimension {dim} exceeds the tensor rank.");
            }
            else if (dimensions[dim] != 1)
            {
                throw new ArgumentException($"Dimension {dim} must be of size 1 to broadcast.");
            }
            else
            {
                var dims = new int[Rank];
                ArrayUtilities.UnsafeCopy(dimensions, ref dims);
                dims[dim] = size;
                return new BroadcastedTensor<T>(this, dims, new int[] {dim});
            }
        }

        public Tensor<T> PadLeft() => InsertDim(0);

        public Tensor<T> PadRight() => InsertDim(Rank);

        public Tensor<T> Reshape(params int[] dims) => Reshape((ReadOnlySpan<int>)dims);
        #endregion

        #region Static methods
        /// <summary>
        /// Compares two tensors by shape and then element by element in flat order.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static int Compare(Tensor<T> left, Tensor<T> right)
        {
            return StructuralComparisons.StructuralComparer.Compare(left, right);
        }

        /// <summary>
        /// Reports whether two tensors share their shape and every element.
        /// </summary>
        /// <param name="left"></param>
        /// <param name="right"></param>
        /// <returns></returns>
        public static bool Equals(Tensor<T> left, Tensor<T> right)
        {
            return StructuralComparisons.StructuralEqualityComparer.Equals(left, right);
        }
        #endregion

        #region IEnumerable members
        IEnumerator IEnumerable.GetEnumerator()
        {
            return ((IEnumerable<T>)this).GetEnumerator();
        }
        #endregion

        #region ICollection members
        int ICollection.Count => (int)Length;

        bool ICollection.IsSynchronized => false;

        object ICollection.SyncRoot => this;

        void ICollection.CopyTo(Array array, int index)
        {
            if (array is T[] destinationArray)
            {
                CopyTo(destinationArray, index);
            }
            else
            {
                if (array == null)
                {
                    throw new ArgumentNullException(nameof(array));
                }
                if (array.Rank != 1)
                {
                    throw new ArgumentException("Only single dimensional arrays are supported for the requested action.", nameof(array));
                }
                if (array.Length < index + Length)
                {
                    throw new ArgumentException("The number of elements in the Tensor is greater than the available space from index to the end of the destination array.", nameof(array));
                }

                for (int i = 0; i < length; i++)
                {
                    array.SetValue(GetValue(i), index + i);
                }
            }
        }
        #endregion

        #region IList members
        object? IList.this[int index]
        {
            get
            {
                return GetValue(index);
            }
            set
            {
                try
                {
                    SetValue(index, value is null ? default(T) : (T)value);
                }
                catch (InvalidCastException)
                {
                    throw new ArgumentException($"The value \"{value}\" is not of type \"{typeof(T)}\" and cannot be used in this generic collection.");
                }
            }
        }

        /// <summary>
        /// Fixed-size collection view.
        /// </summary>
        /// <value>Always true.</value>
        public bool IsFixedSize => true;

        /// <summary>
        /// Writable collection view.
        /// </summary>
        /// <value>Always false.</value>
        public bool IsReadOnly => false;

        int IList.Add(object? value)
        {
            throw new InvalidOperationException();
        }

        void IList.Clear()
        {
            Fill(default(T));
        }

        bool IList.Contains(object? value)
        {
            if (IsCompatibleObject(value))
            {
                return Contains((T)value);
            }
            return false;
        }

        int IList.IndexOf(object? value)
        {
            if (IsCompatibleObject(value))
            {
                return IndexOf((T)value);
            }
            return -1;
        }

        void IList.Insert(int index, object? value)
        {
            throw new InvalidOperationException();
        }

        void IList.Remove(object? value)
        {
            throw new InvalidOperationException();
        }

        void IList.RemoveAt(int index)
        {
            throw new InvalidOperationException();
        }
        #endregion

        #region IEnumerable<T> members
        IEnumerator<T> IEnumerable<T>.GetEnumerator()
        {
            for (int i = 0; i < Length; i++)
            {
                yield return GetValue(i);
            }
        }
        #endregion

        #region ICollection<T> members
        int ICollection<T>.Count => (int)Length;

        void ICollection<T>.Add(T item)
        {
            throw new InvalidOperationException();
        }

        void ICollection<T>.Clear()
        {
            Fill(default(T));
        }

        bool ICollection<T>.Contains(T item)
        {
            return Contains(item);
        }

        /// <summary>
        /// Reports whether an element occurs in flat order.
        /// </summary>
        /// <param name="item">
        /// The object to locate in the Tensor&lt;T&gt;. The value can be null for reference types.
        /// </param>
        /// <returns>
        /// true if item is found in the Tensor&lt;T&gt;; otherwise, false.
        /// </returns>
        protected virtual bool Contains(T item)
        {
            return Length != 0 && IndexOf(item) != -1;
        }

        void ICollection<T>.CopyTo(T[] array, int arrayIndex)
        {
            CopyTo(array, arrayIndex);
        }

        /// <summary>
        /// Copies the elements in flat order into an array at the given position.
        /// </summary>
        /// <param name="array">
        /// The one-dimensional Array that is the destination of the elements copied from Tensor&lt;T&gt;. The Array must have zero-based indexing.
        /// </param>
        /// <param name="arrayIndex">
        /// The zero-based index in array at which copying begins.
        /// </param>
        protected virtual void CopyTo(T[] array, int arrayIndex)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }
            if (array.Length < arrayIndex + Length)
            {
                throw new ArgumentException("The number of elements in the Tensor is greater than the available space from index to the end of the destination array.", nameof(array));
            }

            for (int i = 0; i < length; i++)
            {
                array[arrayIndex + i] = GetValue(i);
            }
        }

        /// <summary>
        /// True when both tensors ultimately read and write the same backing
        /// array (views, reshapes and broadcasts of shared storage included).
        /// Unknown layouts report shared so callers take the safe path.
        /// </summary>
        protected static bool SharesStorage(Tensor<T> a, Tensor<T> b)
        {
            if (ReferenceEquals(a, b)) return true;
            try
            {
                if (!System.Runtime.InteropServices.MemoryMarshal.TryGetArray(a.Storage, out System.ArraySegment<T> sa) || sa.Array is null) return true;
                if (!System.Runtime.InteropServices.MemoryMarshal.TryGetArray(b.Storage, out System.ArraySegment<T> sb) || sb.Array is null) return true;
                return ReferenceEquals(sa.Array, sb.Array);
            }
            catch (Exception) { return true; }
        }

        /// <summary>
        /// Materializes an independent dense copy through public indexing, so
        /// overlapping copies observe original values (memmove semantics).
        /// </summary>
        protected static DenseTensor<T> Snapshot(Tensor<T> source)
        {
            var dims = source.Dimensions.ToArray();
            var tmp = new DenseTensor<T>(dims);
            foreach (var index in source.GetDimensionsIterator())
            {
                tmp[index] = source[index];
            }
            return tmp;
        }

        /// <summary>
        /// Copies values into this tensor. Overlapping source and destination
        /// behave as if copied through a temporary: no corruption.
        /// </summary>
        protected virtual void CopyFrom(Tensor<T> from)
        {
            if (from is null) throw new ArgumentNullException(nameof(from));
            if (!dimensions.SequenceEqual(from.dimensions))
                throw new ArgumentException("The shape of the from tensor is not the same as this tensor.");
            if (SharesStorage(this, from)) from = Snapshot(from);

            
            foreach (var index in from.GetDimensionsIterator())
            {
                this[index] = from[index];
            }
            
        }
        bool ICollection<T>.Remove(T item)
        {
            throw new InvalidOperationException();
        }
        #endregion

        #region IReadOnlyCollection<T> members

        int IReadOnlyCollection<T>.Count => (int)Length;

        #endregion

        #region IList<T> members
        T IList<T>.this[int index]
        {
            get { return GetValue(index); }
            set { SetValue(index, value); }
        }

        int IList<T>.IndexOf(T item)
        {
            return IndexOf(item);
        }

        /// <summary>
        /// Finds the first flat position holding a specific item.
        /// </summary>
        /// <param name="item">Element to locate.</param>
        /// <returns>The index of item if found in the tensor; otherwise, -1.</returns>
        protected virtual int IndexOf(T item)
        {
            for (int i = 0; i < Length; i++)
            {
                if (GetValue(i).Equals(item))
                {
                    return i;
                }
            }

            return -1;
        }

        void IList<T>.Insert(int index, T item)
        {
            throw new InvalidOperationException();
        }

        void IList<T>.RemoveAt(int index)
        {
            throw new InvalidOperationException();
        }
        #endregion

        #region IReadOnlyList<T> members

        T IReadOnlyList<T>.this[int index] => GetValue(index);

        #endregion

        #region IStructuralComparable members
        int IStructuralComparable.CompareTo(object? other, IComparer comparer)
        {
            if (other == null)
            {
                return 1;
            }

            if (other is Tensor<T>)
            {
                return CompareTo((Tensor<T>)other, comparer);
            }

            var otherArray = other as Array;

            if (otherArray != null)
            {
                return CompareTo(otherArray, comparer);
            }

            throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} to {other.GetType()}.", nameof(other));
        }

        private int CompareTo(Tensor<T> other, IComparer comparer)
        {
            if (Rank != other.Rank)
            {
                throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} with Rank {Rank} to {nameof(other)} with Rank {other.Rank}.", nameof(other));
            }

            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] != other.dimensions[i])
                {
                    throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)}s with differning dimension {i}, {dimensions[i]} != {other.dimensions[i]}.", nameof(other));
                }
            }

            int result = 0;

            if (IsReversedStride == other.IsReversedStride)
            {
                for (int i = 0; i < Length; i++)
                {
                    result = comparer.Compare(GetValue(i), other.GetValue(i));
                    if (result != 0)
                    {
                        break;
                    }
                }
            }
            else
            {
                var indices = Rank < ArrayUtilities.StackallocMax ? stackalloc int[Rank] : new int[Rank];
                for (int i = 0; i < Length; i++)
                {
                    ArrayUtilities.GetIndices(strides, IsReversedStride, i, indices, 0);
                    result = comparer.Compare(this[indices], other[indices]);
                    if (result != 0)
                    {
                        break;
                    }
                }
            }

            return result;
        }

        private int CompareTo(Array other, IComparer comparer)
        {
            if (Rank != other.Rank)
            {
                throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} with Rank {Rank} to {nameof(Array)} with rank {other.Rank}.", nameof(other));
            }

            for (int i = 0; i < dimensions.Length; i++)
            {
                var otherDimension = other.GetLength(i);
                if (dimensions[i] != otherDimension)
                {
                    throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} to {nameof(Array)} with differning dimension {i}, {dimensions[i]} != {otherDimension}.", nameof(other));
                }
            }

            int result = 0;
            var indices = new int[Rank];
            for (int i = 0; i < Length; i++)
            {
                ArrayUtilities.GetIndices(strides, IsReversedStride, i, indices, 0);

                result = comparer.Compare(GetValue(i), other.GetValue(indices));

                if (result != 0)
                {
                    break;
                }
            }

            return result;
        }
        #endregion

        #region IStructuralEquatable members
        bool IStructuralEquatable.Equals(object? other, IEqualityComparer comparer)
        {
            if (other == null)
            {
                return false;
            }

            if (other is Tensor<T>)
            {
                return Equals((Tensor<T>)other, comparer);
            }

            var otherArray = other as Array;

            if (otherArray != null)
            {
                return Equals(otherArray, comparer);
            }

            throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} to {other.GetType()}.", nameof(other));
        }

        private bool Equals(Tensor<T> other, IEqualityComparer comparer)
        {
            if (Rank != other.Rank)
            {
                throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} with Rank {Rank} to {nameof(other)} with Rank {other.Rank}.", nameof(other));
            }

            for (int i = 0; i < dimensions.Length; i++)
            {
                if (dimensions[i] != other.dimensions[i])
                {
                    throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)}s with differning dimension {i}, {dimensions[i]} != {other.dimensions[i]}.", nameof(other));
                }
            }

            if (IsReversedStride == other.IsReversedStride)
            {
                for (int i = 0; i < Length; i++)
                {
                    if (!comparer.Equals(GetValue(i), other.GetValue(i)))
                    {
                        return false;
                    }
                }
            }
            else
            {
                var indices = Rank < ArrayUtilities.StackallocMax ? stackalloc int[Rank] : new int[Rank];
                for (int i = 0; i < Length; i++)
                {
                    ArrayUtilities.GetIndices(strides, IsReversedStride, i, indices, 0);

                    if (!comparer.Equals(this[indices], other[indices]))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        private bool Equals(Array other, IEqualityComparer comparer)
        {
            if (Rank != other.Rank)
            {
                throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} with Rank {Rank} to {nameof(Array)} with rank {other.Rank}.", nameof(other));
            }

            for (int i = 0; i < dimensions.Length; i++)
            {
                var otherDimension = other.GetLength(i);
                if (dimensions[i] != otherDimension)
                {
                    throw new ArgumentException($"Cannot compare {nameof(Tensor<T>)} to {nameof(Array)} with differning dimension {i}, {dimensions[i]} != {otherDimension}.", nameof(other));
                }
            }

            var indices = new int[Rank];
            for (int i = 0; i < Length; i++)
            {
                ArrayUtilities.GetIndices(strides, IsReversedStride, i, indices, 0);

                if (!comparer.Equals(GetValue(i), other.GetValue(indices)))
                {
                    return false;
                }
            }

            return true;
        }
        int IStructuralEquatable.GetHashCode(IEqualityComparer comparer)
        {
            int hashCode = 0;
            // Shape is deliberately excluded: equal content hashes equal, and shape
            // differences merely collide.
            for (int i = 0; i < Length; i++)
            {
                hashCode ^= comparer.GetHashCode(GetValue(i));
            }

            return hashCode;
        }
        #endregion

        #region Translations

        /// <summary>
        /// Copies this tensor into dense storage (a Clone when already dense).
        /// </summary>
        /// <returns></returns>
        public virtual DenseTensor<T> ToDenseTensor()
        {
            var denseTensor = new DenseTensor<T>(Dimensions, IsReversedStride);
            foreach (var index in denseTensor.GetDimensionsIterator())
            {
                denseTensor[index] = this[index];
            }
            return denseTensor;
        }

        #endregion

        #region Display and Description
        /// <summary>
        /// Renders the tensor contents as nested brackets
        /// </summary>
        /// <param name="includeWhitespace"></param>
        /// <returns></returns>
        public string PrintData(bool includeWhitespace)
        {
            if (Rank == 0)
            {
                return ((object?)this.GetValue(0))?.ToString() ?? ""; 
            }
            var text = new StringBuilder();

            var strides = ArrayUtilities.GetStrides(dimensions);
            var coords = new int[Rank];
            var lastAxis = Rank - 1;
            var rowLength = dimensions[lastAxis];
            var rowCount = Length / rowLength;

            int depth = 0;
            for (int row = 0; row < Length; row += rowLength)
            {
                ArrayUtilities.GetIndices(strides, false, row, coords, 0);

                while ((depth < lastAxis) && (coords[depth] == 0))
                {
                    // start up
                    if (includeWhitespace)
                    {
                        Pad(text, depth, 4);
                    }
                    depth++;
                    text.Append('[');
                    if (includeWhitespace)
                    {
                        text.AppendLine();
                    }
                }

                for (int cell = 0; cell < rowLength; cell++)
                {
                    coords[lastAxis] = cell;

                    if ((cell == 0))
                    {
                        if (includeWhitespace)
                        {
                            Pad(text, depth, 4);
                        }
                        text.Append('[');
                    }
                    else
                    {
                        text.Append(',');
                    }
                    if (ElementType == TensorElementType.Float || ElementType == TensorElementType.Double)
                    {
                        text.Append(string.Format("{0:0.00000}", this[coords]));
                    }
                    else
                    {
                        text.Append(this[coords]);
                    }
                }
                text.Append(']');

                for (int i = Rank - 2; i >= 0; i--)
                {
                    var final = dimensions[i] - 1;
                    if (coords[i] == final)
                    {
                        // close out
                        --depth;
                        if (includeWhitespace)
                        {
                            text.AppendLine();
                            Pad(text, depth, 4);
                        }
                        text.Append(']');
                    }
                    else
                    {
                        text.Append(',');
                        if (includeWhitespace)
                        {
                            text.AppendLine();
                        }
                        break;
                    }
                }
            }

            return text.ToString();

            void Pad(StringBuilder text, int levels, int width)
            {
                for (int level = 0; level < levels; level++)
                {
                    for (int s = 0; s < width; s++)
                    {
                        text.Append(' ');
                    }
                }
            }
        }

        public string PrintShape() => "[" + string.Join(',', dimensions) + "]";
        
        public Tensor<T> WithName(string name)
        {
            this.Name = name;
            return this;
        }
        #endregion

        #region ITensor members
        public string Name { get; set; } = "";

        public TensorElementType ElementType { get; } = GetTypeInfo(typeof(T))?.ElementType ?? throw new NotSupportedException($"Tensor element type {typeof(T).Name} is not supported.");

        public Type PrimitiveType { get; } = typeof(T);

        /// <summary>
        /// A copy of the shape metadata. Mutating the result never affects
        /// this tensor; use <see cref="Dimensions"/> for an allocation-free view.
        /// </summary>
        int[] ITensor.Dims => (int[])this.dimensions.Clone();

        ITensor ITensor.Clone() => Clone();

        ITensor ITensor.CloneEmpty() => CloneEmpty();

        INumericTensor INumericTensor.CloneEmpty<U>() => CloneEmpty<U>();

        INumericTensor INumericTensor.Reshape(int[] shape) => this.Reshape(shape);

        INumericTensor INumericTensor.InsertDim(int dim) => this.InsertDim(dim);

        INumericTensor INumericTensor.RemoveDim(int dim) => this.RemoveDim(dim);

        INumericTensor INumericTensor.BroadcastDim(int dim, int size) => this.BroadcastDim(dim, size);

        INumericTensor INumericTensor.ToDenseTensor() => this.ToDenseTensor();

        ITensor  ITensor.this[params object[] indices]
        {
            get => new TensorSlice<T>(this, ExpandEllipsis(ToSliceIndexes(indices)));
            set
            {
                var ts = new TensorSlice<T>(this, ExpandEllipsis(ToSliceIndexes(indices)));
                ts.CopyFrom((Tensor<T>) value);
            }
        }

        /// <summary>
        /// Converts untyped slice coordinates into slice selectors, one per position.
        /// </summary>
        private static SliceIndex[] ToSliceIndexes(object[] indices)
        {
            var selectors = new SliceIndex[indices.Length];
            for (int i = 0; i < indices.Length; i++) selectors[i] = SliceIndex.FromObj(indices[i]);
            return selectors;
        }

        object ITensor.this[params int[] indices]
        {
            get => this[indices];
            set => this[indices] = (T) value;
        }

        object ITensor.GetValue(int index) => this.GetValue(index); 

        void ITensor.SetValue(int index, object? value) => this.SetValue(index, value is null ? default(T) : (T)value);

        INumericTensor INumericTensor.Slice(string indices) => new TensorSlice<T>(this, ExpandEllipsis(SliceIndex.ParseSlices(indices)));

        Array ITensor.ToArray() => this.ToArray();
        #endregion

        #region Slicing
        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public int[] SliceAxes(params SliceIndex[] input_slices)
        {
            if (dimensions is null || dimensions.Length == 0)
                throw new InvalidOperationException("Unable to slice an empty shape.");

            int len = this is TensorSlice<T> view ? this.Rank + view.parent.Rank : this.Rank;
            // Scratch spans sized for the merged worst case; every write below is bounds-checked.
            Span<SliceDef> defs = stackalloc SliceDef[len];
            Span<int> extents = stackalloc int[len];
            int ndefs = 0;
            for (int i = 0; i < dimensions.Length; i++)
            {
                if (ndefs >= len) throw new ArgumentOutOfRangeException(nameof(input_slices), "Too many slice selectors for this tensor shape.");
                var dim = dimensions[i];
                var slice = input_slices.Length > i ? input_slices[i] : SliceIndex.All; //fill missing selectors
                var def = slice.ToSliceDef(dim);
                defs[ndefs] = def;
                extents[ndefs] = Math.Abs(defs[ndefs].Count); // index selectors report -1 but keep one element.
                ndefs++;
            }

            if (this is TensorSlice<T> outer)
            {
                // Fold the new selectors into the existing view, reinserting parent axes
                // that an earlier index selector had reduced away.
                for (int i = 0; i < outer.parent.Rank; i++)
                {
                    var prior = outer.slices[i];
                    if (prior.IsIndex)
                    {
                        if (ndefs >= len) throw new ArgumentOutOfRangeException(nameof(input_slices), "Too many slice selectors for this tensor shape.");
                        for (int j = ndefs; j > i; j--) { defs[j] = defs[j - 1]; extents[j] = extents[j - 1]; }
                        defs[i] = prior;
                        extents[i] = 1;
                        ndefs++;
                        continue;
                    }

                    defs[i] = outer.slices[i].Merge(defs[i]);
                    extents[i] = Math.Abs(defs[i].Count);
                }
            }

            int kept = 0;
            for (int i = 0; i < ndefs; i++) if (!defs[i].IsIndex) kept++;
            var shape = new int[kept];
            kept = 0;
            for (int i = 0; i < ndefs; i++) if (!defs[i].IsIndex) { shape[kept] = extents[i]; kept++; }
            return shape;
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
        public SliceIndex[] ExpandEllipsis(SliceIndex[] slices)
        {
            bool hasEllipsis = false;
            foreach (var selector in slices)
            {
                if (selector.IsEllipsis) { hasEllipsis = true; break; }
            }
            if (!hasEllipsis)
            {
                return slices;
            }
            if (slices.Length == 1)
            {
                var all = new SliceIndex[Rank];
                Array.Fill(all, SliceIndex.All);
                return all;
            }
            // Axes already covered without counting ellipsis or new-axis markers.
            int covered = 0;
            foreach (var selector in slices)
            {
                if (selector.IsNewAxis || selector.IsEllipsis)
                    continue;
                covered++;
            }
            var expanded = new List<SliceIndex>();
            foreach (var selector in slices)
            {
                if (selector.IsEllipsis)
                {
                    for (int i = 0; i < dimensions.Length - covered; i++)
                        expanded.Add(SliceIndex.All);
                    continue;
                }

                expanded.Add(selector);
            }
            return expanded.ToArray();
        }

        /// <summary>
        ///  Converts a flat row-major position back into coordinates (slicing is ignored).
        ///  Example: Shape (2,3)
        /// 0 => [0, 0]
        /// 1 => [0, 1]
        /// ...
        /// 5 => [1, 2]
        /// </summary>
        /// <param name="offset">the index if you would iterate from 0 to shape.size in row major order</param>
        /// <returns></returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public int[] GetCoordinates(int offset)
        {
            var coords = new int[strides.Length];
            int rest = offset;
            for (int i = 0; i < strides.Length; i++)
            {
                unchecked
                {
                    int step = strides[i];
                    if (step == 0)
                    {
                        coords[i] = 0;
                    }
                    else
                    {
                        coords[i] = rest / step;
                        rest -= coords[i] * step;
                    }
                }
            }

            return coords;
        }

        public TensorSlice<T> Slice(params SliceIndex[] indices) => new TensorSlice<T>(this, ExpandEllipsis(indices));
        #endregion

        #region Storage
        /// <summary>
        /// Root backing storage for dense data. Views resolve to their ultimate
        /// dense owner (slices to the parent chain, broadcasts to the source).
        /// Non-dense views are densified before kernels pin this storage.
        /// </summary>
        public Memory<T> Storage => this switch
        {
            DenseTensor<T> dt => dt.Buffer,
            BroadcastedTensor<T> bt => bt.source.Storage,
            TensorSlice<T> ts => ts.parent.Storage,
            _ => throw new NotImplementedException("Storage property not implemented for this tensor type.")
        };

        /// <summary>
        /// Maps logical coordinates to an offset in <see cref="Storage"/> using
        /// each level standard dense strides (broadcasts via effective strides,
        /// slices via parent-relative offsets). Exact for dense-compatible
        /// layouts, which is all kernels ever pass after densification.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public int GetStorageIndex(int[] indices) => this switch
        {
            DenseTensor<T> dt => ArrayUtilities.GetIndex(dt.strides, indices),
            BroadcastedTensor<T> bt => ArrayUtilities.GetIndex(bt.effectiveStrides, indices),
            TensorSlice<T> ts => ts.GetOffset(indices),
            _ => throw new NotSupportedException("GetStorageIndex method not implemented for this tensor type.")

        };
        #endregion

        #region Dimensions iterator
        public bool ShapeEquals(Tensor<T> t) => this.dimensions.SequenceEqual(t.dimensions);
        
        public TensorDimensionsIterator GetDimensionsIterator(Range r) => new TensorDimensionsIterator(dimensions[r]);

        public TensorDimensionsIterator GetDimensionsIterator() => GetDimensionsIterator(..);
        #endregion

        private static bool IsCompatibleObject([NotNullWhen(true)] object? value)
        {
            // Only values of exactly T are compatible; null never qualifies for value types.
            return value is T;
        }

        public static Tensor<int> Arange(int start, int stop) => Arange(start, stop, 1);

        public static Tensor<int> Arange(int start, int stop, int step)
        {
            if (step == 0)
                throw new ArgumentException("step can't be 0", nameof(step));

            // Negative steps walk the same span downwards, so normalize to an
            // ascending span first and mirror the fill below.
            int lo = start;
            int hi = stop;
            int stride = step;
            bool descending = false;
            if (stride < 0)
            {
                descending = true;
                stride = Math.Abs(stride);
                lo = stop;
                hi = start;
            }

            if (lo > hi)
                throw new Exception("parameters invalid, start is greater than stop.");

            int length = (int)Math.Ceiling((hi - lo + 0.0d) / stride);
            var sequence = new DenseTensor<int>((ReadOnlySpan<int>)new int[] { length });

            if (descending)
            {
                for (int add = length - 1, i = 0; add >= 0; add--, i++)
                    sequence[i] = 1 + lo + add * stride;
            }
            else
            {
                for (int i = 0; i < length; i++)
                    sequence[i] = lo + i * stride;
            }

            return sequence;
        }

        public static Tensor<float> Arange(float start, float stop) => Arange(start, stop, 1.0f);

        public static Tensor<float> Arange(float start, float stop, float step)
        {
            if (step == 0.0f)
                throw new ArgumentException("step can't be 0", nameof(step));

            // Negative steps walk the same span downwards, so normalize to an
            // ascending span first and mirror the fill below.
            float lo = start;
            float hi = stop;
            float stride = step;
            bool descending = false;
            if (stride < 0.0f)
            {
                descending = true;
                stride = Math.Abs(stride);
                lo = stop;
                hi = start;
            }

            if (lo > hi)
                throw new Exception("parameters invalid, start is greater than stop.");

            int length = (int)Math.Ceiling((hi - lo + 0.0d) / stride);
            var sequence = new DenseTensor<float>((ReadOnlySpan<int>)new int[] { length });

            if (descending)
            {
                for (int add = length - 1, i = 0; add >= 0; add--, i++)
                    sequence[i] = 1.0f + lo + add * stride;
            }
            else
            {
                for (int i = 0; i < length; i++)
                    sequence[i] = lo + i * stride;
            }

            return sequence;
        }
        public static Tensor<T> Zeros(params int[] dims)
        {
            var t = new DenseTensor<T>((ReadOnlySpan<int>)dims);
            t.Fill(Zero);
            return t;
        }

        public static Tensor<T> Ones(params int[] dims)
        {
            var t = new DenseTensor<T>((ReadOnlySpan<int>)dims);
            t.Fill(One);
            return t;
        }

        public static Tensor<float> Rand(params int[] dims)
        {
            var t = new DenseTensor<float>((ReadOnlySpan<int>)dims);
            var rnd = new Random();
            for (int i = 0; i < t.Length; i++)
            {
                t.SetValue(i, rnd.NextSingle());
            }
            return t;
        }

        public static Tensor<int> RandN(params int[] dims)
        {
            var t = new DenseTensor<int>((ReadOnlySpan<int>)dims);
            var rnd = new Random();
            for (int i = 0; i < t.Length; i++)
            {
                t.SetValue(i, rnd.Next());
            }
            return t;
        }
    }
}
