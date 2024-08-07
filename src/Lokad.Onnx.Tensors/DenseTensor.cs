﻿// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is copied and adapted from the following git repository -
// https://github.com/dotnet/corefx
// Commit ID: bdd0814360d4c3a58860919f292a306242f27da1
// Path: /src/System.Numerics.Tensors/src/System/Numerics/Tensors/DenseTensor.cs
// Original license statement below -

// Licensed to the .NET Foundation under one or more agreements.
// The .NET Foundation licenses this file to you under the MIT license.
// See the LICENSE file in the project root for more information.

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
    /// Represents a multi-dimensional collection of objects of type T that can be accessed by indices.  
    /// DenseTensor stores values in a contiguous sequential block of memory where all values are represented.
    /// </summary>
    /// <typeparam name="T">
    /// Type contained within the Tensor. Typically a value type such as int, double, float, etc.
    /// </typeparam>
    public unsafe class DenseTensor<T> : Tensor<T> where T :  unmanaged
    {
        #region Fields
        protected readonly ArraySegment<T> arr;
        protected readonly Memory<T> memory;
        #endregion

        #region Properties
        /// <summary>
        /// Memory storing backing values of this tensor.
        /// </summary>
        public Memory<T> Buffer => memory;
        #endregion

        #region Constructors
        internal DenseTensor(Array fromArray, bool reverseStride = false) : base(fromArray, reverseStride)
        {
            // copy initial array
            var backingArray = new T[fromArray.Length];
            
            int index = 0;
            if (reverseStride)
            {
                // Array is always row-major
                var sourceStrides = ArrayUtilities.GetStrides(dimensions);

                foreach (var item in fromArray)
                {
                    var destIndex = ArrayUtilities.TransformIndexByStrides(index++, sourceStrides, false, strides);
                    backingArray[destIndex] = (T)item;
                }
            }
            else
            {
                foreach (var item in fromArray)
                {
                    backingArray[index++] = (T)item;
                }
            }

            arr = backingArray;
            memory = backingArray;
        }

        /// <summary>
        /// Initializes a rank-1 Tensor using the specified <paramref name="length"/>.
        /// </summary>
        /// <param name="length">Size of the 1-dimensional tensor</param>
        public DenseTensor(int length) : base(length)
        {
            memory = new T[length];
        }

        /// <summary>
        /// Initializes a rank-n Tensor using the dimensions specified in <paramref name="dimensions"/>.
        /// </summary>
        /// <param name="dimensions">
        /// An span of integers that represent the size of each dimension of the DenseTensor to create.
        /// </param>
        /// <param name="reverseStride">
        /// False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension 
        /// is most minor (closest together): akin to row-major in a rank-2 tensor.  
        /// True to indicate that the last dimension is most major (farthest apart) and the first dimension is most 
        /// minor (closest together): akin to column-major in a rank-2 tensor.
        /// </param>
        public DenseTensor(ReadOnlySpan<int> dimensions, bool reverseStride = false) : base(dimensions, reverseStride)
        {
            arr = new T[Length];
            memory = arr;
        }

        /// <summary>
        /// Constructs a new DenseTensor of the specified dimensions, wrapping existing backing memory for the contents.
        /// </summary>
        /// <param name="memory"></param>
        /// <param name="dimensions">
        /// An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <param name="reverseStride">
        /// False (default) to indicate that the first dimension is most major (farthest apart) and the last dimension 
        /// is most minor (closest together): akin to row-major in a rank-2 tensor.  
        /// True to indicate that the last dimension is most major (farthest apart) and the first dimension is most 
        /// minor (closest together): akin to column-major in a rank-2 tensor.
        /// </param>
        public DenseTensor(Memory<T> memory, ReadOnlySpan<int> dimensions, bool reverseStride = false) 
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
        /// Gets the value at the specified index, where index is a linearized version of n-dimension indices 
        /// using strides. For a scalar, use index = 0
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <returns>The value at the specified position in this Tensor.</returns>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public override T GetValue(int index)
        {
            return arr[index];   
        }

        /// <summary>
        /// Sets the value at the specified index, where index is a linearized version of n-dimension indices 
        /// using strides. For a scalar, use index = 0
        /// </summary>
        /// <param name="index">An integer index computed as a dot-product of indices.</param>
        /// <param name="value">The new value to set at the specified position in this Tensor.</param>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public override void SetValue(int index, T value)
        {
            arr[index] = value;
        }

        /// <summary>
        /// Overrides Tensor.CopyTo(). Copies the content of the Tensor
        /// to the specified array starting with arrayIndex
        /// </summary>
        /// <param name="array">destination array</param>
        /// <param name="arrayIndex">start index</param>
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
        /// Determines the index of a specific item in the Tensor&lt;T&gt;.
        /// </summary>
        /// <param name="item">Object to locate</param>
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
        /// Creates a shallow copy of this tensor, with new backing storage.
        /// </summary>
        /// <returns>A shallow copy of this tensor.</returns>
        public override Tensor<T> Clone()
        {
            // create copy
            var memory = new T[Length];
            this.memory.CopyTo(memory);
            return new DenseTensor<T>(memory, dimensions, IsReversedStride);
        }

        /// <summary>
        /// Creates a new Tensor of a different type with the specified dimensions and the same layout as this tensor 
        /// with elements initialized to their default value.
        /// </summary>
        /// <typeparam name="TResult">Type contained in the returned Tensor.</typeparam>
        /// <param name="dimensions">
        /// An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor with the same layout as this tensor but different type and dimensions.</returns>
        public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> dimensions)
        {
            return new DenseTensor<TResult>(dimensions, IsReversedStride);
        }
        
        /// <summary>
        /// Reshapes the current tensor to new dimensions, using the same backing storage.
        /// </summary>
        /// <param name="dimensions">
        /// An span of integers that represent the size of each dimension of the DenseTensor to create.</param>
        /// <returns>A new tensor that reinterprets backing Buffer of this tensor with different dimensions.</returns>
        public override Tensor<T> Reshape(ReadOnlySpan<int> dimensions)
        {
 
            var newSize = ArrayUtilities.ComputeOffsetForReduction(dimensions);

            if (newSize != Length)
            {
                throw new ArgumentException($"Cannot reshape array due to mismatch in lengths, currently {Length} would become {newSize}.", nameof(dimensions));
            }

            return new DenseTensor<T>(Buffer, dimensions, IsReversedStride);
        }

        protected override void CopyFrom(Tensor<T> from)
        {
            if (from is DenseTensor<T> d)
            {
                var handle = memory.Pin();
                var handle2 = d.memory.Pin();
                var ptr = (T*) handle.Pointer;
                var ptr2 = (T*)handle2.Pointer;
                for (int i = 0; i < from.Length; i++)
                {
                    ptr[i] = ptr2[i];
                }
                handle.Dispose();   
                handle2.Dispose();
            }
            else
            {
                foreach (var index in from.GetDimensionsIterator())
                {
                    this[index] = from[index];
                }
            }
        }

        public override DenseTensor<T> ToDenseTensor() => this;

        public static DenseTensor<T> OfShape(params int[] dims) => new DenseTensor<T>((ReadOnlySpan<int>) dims);
        #endregion

        #region Static methods
        public static DenseTensor<T> OfValues(Array data) => data.ToTensor<T>();

        public static DenseTensor<T> Scalar(T value) => new DenseTensor<T>(new T[1] { value }, Array.Empty<int>());
        #endregion
    }
}
