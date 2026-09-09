using System.Runtime.InteropServices;
using System;
using System.Buffers;
using System.Linq;
using System.Runtime.Versioning;
using System.Reflection.Metadata.Ecma335;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx
{
    /// <summary>
    /// A tensor whose logical contents live in one contiguous block of memory in row-major order.
    /// Independently implemented from the layout contract: flat index is the dot product of
    /// coordinates and strides, and every element is represented exactly once.
    /// </summary>
    /// <typeparam name="T">
    /// The element type, always an unmanaged value type in this library.
    /// </typeparam>
    public unsafe class DenseTensor<T> : Tensor<T> where T :  unmanaged
    {
        #region Fields
        protected readonly ArraySegment<T> arr;
        protected readonly Memory<T> memory;
        #endregion

        #region Properties
        /// <summary>
        /// The live backing store viewed with this tensor's dimensions and strides.
        /// </summary>
        public Memory<T> Buffer => memory;

        /// <summary>
        /// The caller-visible array only when this value densely owns exactly its logical
        /// contents in one zero-based array; views and slices never qualify, so storage
        /// owned elsewhere cannot leak out through this probe.
        /// </summary>
        internal override Array? OwnedBufferArray()
        {
            if (memory.Length != Length) return null;
            if (!MemoryMarshal.TryGetArray(memory, out ArraySegment<T> window) || window.Array is null) return null;
            if (window.Offset != 0 || window.Array.Length != memory.Length) return null;
            return window.Array;
        }
        #endregion

        #region Constructors
        internal DenseTensor(Array fromArray, bool reverseStride) : base(fromArray, reverseStride)
        {
            // Fill a fresh row-major-scale backing array; the source keeps its contents.
            var backingArray = new T[fromArray.Length];
            if (reverseStride)
            {
                // Incoming arrays always enumerate in row-major order, so each element
                // is placed through the stride remap into this tensor's layout.
                var rowMajorStrides = ArrayUtilities.GetStrides(dimensions);
                int source = 0;
                foreach (var item in fromArray)
                {
                    backingArray[ArrayUtilities.TransformIndexByStrides(source++, rowMajorStrides, false, strides)] = (T)item;
                }
            }
            else if (fromArray.GetType().GetElementType() == typeof(T))
            {
                // Same-type rectangular arrays already lie contiguous in memory for any
                // rank, so one block copy moves them with no per-element boxing.
                MemoryMarshal.CreateReadOnlySpan(
                    ref Unsafe.As<byte, T>(ref MemoryMarshal.GetArrayDataReference(fromArray)),
                    fromArray.Length).CopyTo(backingArray);
            }
            else
            {
                int flat = 0;
                foreach (var item in fromArray)
                {
                    backingArray[flat++] = (T)item;
                }
            }

            arr = backingArray;
            memory = backingArray;
        }

        /// <summary>
        /// Creates a rank-1 tensor of the given length with default-valued elements.
        /// </summary>
        /// <param name="length">Element count of the one-dimensional tensor.</param>
        public DenseTensor(int length) : base(length)
        {
            arr = new T[length];
            memory = arr;
        }

        /// <summary>
        /// Creates a tensor of the given shape with default-valued elements.
        /// </summary>
        /// <param name="dimensions">
        /// The extent of every axis; the product is the element count.
        /// </param>
        /// <param name="reverseStride">
        /// False (default) for row-major strides, true for reversed strides.
        /// </param>
        public DenseTensor(ReadOnlySpan<int> dimensions, bool reverseStride) : base(dimensions, reverseStride)
        {
            arr = new T[Length];
            memory = arr;
        }

        /// <summary>
        /// Creates a row-major tensor of the specified dimensions.
        /// </summary>
        public DenseTensor(ReadOnlySpan<int> dimensions) : this(dimensions, false)
        {
        }

        /// <summary>
        /// Creates a tensor of the specified dimensions over already-owned backing memory.
        /// </summary>
        /// <param name="memory">Backing store shared with the new tensor.</param>
        /// <param name="dimensions">
        /// The extent of every axis; the product is the element count.</param>
        /// <param name="reverseStride">
        /// False (default) for row-major strides, true for reversed strides.
        /// </param>
        /// <summary>
        /// Creates a row-major tensor over already-owned backing memory.
        /// </summary>
        public DenseTensor(Memory<T> memory, ReadOnlySpan<int> dimensions)
            : this(memory, dimensions, false)
        {
        }

        public DenseTensor(Memory<T> memory, ReadOnlySpan<int> dimensions, bool reverseStride) 
            : base(dimensions, reverseStride)
        {
            if (!MemoryMarshal.TryGetArray<T>(memory, out arr)) throw new InvalidOperationException();
            this.memory = memory;

            if (Length != memory.Length)
            {
                throw new ArgumentException(
                    $"Length of {nameof(memory)} ({memory.Length}) must match product of " +
                    $"{nameof(dimensions)} ({Length}).");
            }
        }
        #endregion

        #region Overrides
        /// <summary>
        /// Reads the element at a flat index (the stride dot product of coordinates).
        /// </summary>
        /// <param name="index">Flat position; 0 addresses a scalar.</param>
        /// <returns>The stored element.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public override T GetValue(int index)
        {
            return arr[index];   
        }

        /// <summary>
        /// Writes the element at a flat index (the stride dot product of coordinates).
        /// </summary>
        /// <param name="index">Flat position; 0 addresses a scalar.</param>
        /// <param name="value">Replacement element.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public override void SetValue(int index, T value)
        {
            arr[index] = value;
        }

        /// <summary>
        /// Copies the logical contents in flat order into the destination array
        /// </summary>
        /// <param name="array">Destination array.</param>
        /// <param name="arrayIndex">First destination position.</param>
        protected override void CopyTo(T[] array, int arrayIndex)
        {
            if (array == null)
            {
                throw new ArgumentNullException(nameof(array));
            }
            if (array.Length < arrayIndex + Length)
            {
                throw new ArgumentException(
                    "The number of elements in the Tensor is greater than the available space from index to " + 
                    "the end of the destination array.", nameof(array));
            }

            Buffer.Span.CopyTo(array.AsSpan(arrayIndex));
        }

        /// <summary>
        /// Finds the first flat position holding a specific item.
        /// </summary>
        /// <param name="item">Element to locate.</param>
        /// <returns>The index of item if found in the tensor; otherwise, -1</returns>
        protected override int IndexOf(T item)
        {
            // TODO: use Span.IndexOf when/if it removes the IEquatable type constraint
            if (MemoryMarshal.TryGetArray<T>(Buffer, out var arraySegment))
            {
                if (arraySegment.Array is null) throw new NullReferenceException(nameof(arraySegment));
                var result = Array.IndexOf(arraySegment.Array, item, arraySegment.Offset, arraySegment.Count);
                if (result != -1)
                {
                    result -= arraySegment.Offset;
                }
                return result;
            }
            else
            {
                return base.IndexOf(item);
            }
        }

        /// <summary>
        /// Copies the elements into fresh backing storage with the same shape.
        /// </summary>
        /// <returns>A new tensor holding a copy of every element.</returns>
        public override Tensor<T> Clone()
        {
            // Duplicate the elements into fresh storage; the shape travels unchanged.
            var copy = new T[Length];
            Buffer.Span.CopyTo(copy);
            return new DenseTensor<T>(copy.AsMemory(), Dimensions, IsReversedStride);
        }

        /// <summary>
        /// Creates an empty tensor of a different element type with the given shape.
        /// </summary>
        /// <typeparam name="TResult">Element type of the new tensor.</typeparam>
        /// <param name="dimensions">
        /// The extent of every axis; the product is the element count.</param>
        /// <returns>A new empty tensor of the requested type and shape.</returns>
        public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        {
            return new DenseTensor<TResult>(dimensions, IsReversedStride);
        }
        
        /// <summary>
        /// Reinterprets the same backing storage under new dimensions.
        /// </summary>
        /// <param name="dimensions">
        /// The extent of every axis; the product is the element count.</param>
        /// <returns>A new tensor over the same backing storage.</returns>
        public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions)
        {
            // Only the shape changes; the element count must be preserved exactly.
            int reshaped = ArrayUtilities.ComputeOffsetForReduction(dimensions, 0);
            if (reshaped != Length)
            {
                throw new ArgumentException($"Cannot reshape array due to mismatch in lengths, currently {Length} would become {reshaped}.", nameof(dimensions));
            }

            return new DenseTensor<T>(Buffer, dimensions, IsReversedStride);
        }

        protected override void CopyFrom(Tensor<T> from)
        {
            // A source sharing this backing store is snapshotted first so the
            // copy observes the original values rather than partially written ones.
            if (SharesStorage(this, from)) from = Snapshot(from);
            if (from is DenseTensor<T> dense)
            {
                dense.memory.Span.CopyTo(memory.Span);
            }
            else
            {
                foreach (var index in from.GetDimensionsIterator())
                {
                    this[index] = from[index];
                }
            }
        }

        public override DenseTensor<T> ToDenseTensor()
        {
            // Row-major values already sit in flat order, so the tensor itself qualifies.
            if (!IsReversedStride)
            {
                return this;
            }

            // Reversed layouts reorder on the way out element by element.
            var rowMajor = new DenseTensor<T>(Dimensions, reverseStride: false);
            foreach (var index in rowMajor.GetDimensionsIterator())
            {
                rowMajor[index] = this[index];
            }

            return rowMajor;
        }

        public static DenseTensor<T> OfShape(params int[] dims) => new DenseTensor<T>((ReadOnlySpan<int>) dims);
        #endregion

        #region Static methods
        public static DenseTensor<T> OfValues(Array data) => data.ToTensor<T>();

        /// <summary>
        /// Copies a single-dimensional array into a new rank-1 tensor with a
        /// block copy and no per-element boxing. The source keeps its contents.
        /// </summary>
        public static DenseTensor<T> OfValues(T[] values)
        {
            if (values is null) throw new ArgumentNullException(nameof(values));
            var output = new DenseTensor<T>(values.Length);
            values.AsSpan().CopyTo(output.Buffer.Span);
            return output;
        }

        /// <summary>
        /// Copies a span into a new tensor of the given dimensions with a block
        /// copy and no per-element boxing. The span length must equal the shape
        /// product. For shared storage, wrap memory with the Memory constructor.
        /// </summary>
        public static DenseTensor<T> OfValues(ReadOnlySpan<T> values, int[] dims)
        {
            if (dims is null) throw new ArgumentNullException(nameof(dims));
            var output = new DenseTensor<T>(dims);
            if (values.Length != (int)output.Length) throw new ArgumentException("Span length must match the product of the dimensions.", nameof(values));
            values.CopyTo(output.Buffer.Span);
            return output;
        }

        public static DenseTensor<T> Scalar(T value) => new DenseTensor<T>(new T[1] { value }, Array.Empty<int>());
        #endregion
    }
}
