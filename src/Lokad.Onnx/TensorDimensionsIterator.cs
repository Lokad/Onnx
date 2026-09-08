using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public struct TensorDimensionsIterator : IEnumerable<int[]>, IEnumerator<int[]>
    {
        #region Constructors
        /// <summary>
        /// Defines the enumeration contract: a scalar (rank 0) enumerates a
        /// single zero coordinate; any zero extent enumerates nothing;
        /// negative extents are rejected. The yielded <see cref="Index"/> array
        /// is borrowed and reused across steps: copy it to retain coordinates.
        /// </summary>
        public TensorDimensionsIterator(int[] dims)
        {
            if (dims is null) throw new ArgumentNullException("Can't construct TensorDimensionsIterator with an empty shape.");
            foreach (var d in dims)
            {
                if (d < 0) throw new ArgumentOutOfRangeException(nameof(dims), "Tensor shape extents must be non-negative.");
            }

            if (dims.Length == 0)
                dims = new int[] { 1 };

            dimensions = dims;
            Index = new int[dims.Length];
            resetto = subcursor = dimensions.Length - 1;
            endCallback = null;
            empty = dims.Any(d => d == 0);
        }

       
        public TensorDimensionsIterator(int[] dims, EndCallbackHandler endCallback) : this(dims)
        {
            this.endCallback = endCallback;
        }
        #endregion

        /// <summary>
        /// Starts an independent cursor over the shared configuration: manual
        /// cursor moves (<see cref="Next"/>) never affect new enumerations, and
        /// pattern-based foreach over the struct avoids boxing entirely.
        /// </summary>
        public TensorDimensionsIterator GetEnumerator()
        {
            if (dimensions is null) return default;
            return new TensorDimensionsIterator(dimensions);
        }

        IEnumerator<int[]> IEnumerable<int[]>.GetEnumerator() => GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>Advances this cursor; false when the sequence is exhausted.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public bool MoveNext()
        {
            if (Index is null || empty) return false;
            if (!moveStart)
            {
                moveStart = true;
                return true;
            }
            else
            {
                return Next() != null;
            }
        }

        bool IEnumerator.MoveNext() => MoveNext();

        object IEnumerator.Current => Index;

        void IDisposable.Dispose() => Reset();
        public void Reset()
        {
            Array.Clear(Index, 0, Index.Length);
            subcursor = resetto;
            moveStart = false;
        }

       
        public int[] Current
        {
            [MethodImpl(MethodImplOptions.AggressiveInlining)]
            get => Index;
        } 
        

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public int[]? Next()
        {
            if (Index is null || empty || subcursor <= -1)
                return null;

            if (++Index[subcursor] >= dimensions[subcursor])
            {
            _repeat:
                Index[subcursor] = 0;

                do
                {
                    if (--subcursor <= -1)
                    {
                        return null;
                    }
                } while (dimensions[subcursor] <= 1);

                ++Index[subcursor];
                if (Index[subcursor] >= dimensions[subcursor])
                    goto _repeat;

                subcursor = resetto;
            }
            
            return Index;
        }

        public TensorDimensionsIterator Append(params int[] dims) => new TensorDimensionsIterator(dimensions.Concat(dims).ToArray());

        public TensorDimensionsIterator this [params int[] indices] => Append(indices);

        public SliceIndex[] AppendEllipsis() => Index.Select(i => new SliceIndex(i)).Append(SliceIndex.Ellipsis).ToArray();

        public SliceIndex[] PrependEllipsis() => Index.Select(i => new SliceIndex(i)).Prepend(SliceIndex.Ellipsis).ToArray();

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public SliceIndex[] AppendSliceIndices(params SliceIndex[] indices) => Index.Select(i => (SliceIndex) i).Concat(indices).ToArray();  

        
        public SliceIndex[] this[params SliceIndex[] indices] => AppendSliceIndices(indices);

        #region Fields
        public delegate void EndCallbackHandler(ref TensorDimensionsIterator incr);
        private readonly EndCallbackHandler? endCallback;
        private readonly int[] dimensions;
        private readonly int resetto;
        private readonly bool empty;
        public readonly int[] Index;
        private int subcursor;
        private bool moveStart = false;
        #endregion

    }

    public struct TensorFixedDimensionsIterator : IEnumerable<int[]>, IEnumerator<int[]>
    {
        public int[] fixedDims;
        public int[] dims;
        public int length;
        public TensorDimensionsIterator iterator;
        public int[] Index;
        public int[] VariableIndex => iterator.Index;

        public TensorFixedDimensionsIterator(int[] fixedDimensions, params int[] dims)
        {
            this.fixedDims = fixedDimensions;
            this.dims = dims;
            this.length = fixedDimensions.Length + dims.Length;
            iterator = new TensorDimensionsIterator(dims);
            Index = new int[length];
            fixedDims.CopyTo(Index, 0);
        }

        public TensorFixedDimensionsIterator(ITensor t, Range r, params int[] dims) : this(t.Dims[r], dims)
        {

        }

        /// <summary>
        /// Starts an independent cursor: the fixed head is copied once per
        /// enumeration and the variable cursor starts fresh, so repeated
        /// enumeration and Reset behave identically.
        /// </summary>
        public TensorFixedDimensionsIterator GetEnumerator()
        {
            if (fixedDims is null || dims is null) return default;
            return new TensorFixedDimensionsIterator(fixedDims, dims);
        }

        IEnumerator<int[]> IEnumerable<int[]>.GetEnumerator() => GetEnumerator();

        IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

        /// <summary>Advances this cursor; false when the sequence is exhausted.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveOptimization)]
        public bool MoveNext()
        {
            if (Index is null) return false;
            if (iterator.MoveNext())
            {
                unchecked
                {
                    for (int i = fixedDims.Length; i < length; i++)
                    {
                        Index[i] = iterator.Index[i - fixedDims.Length];
                    }
                    //iterator.Index.CopyTo(Index, fixedDims.Length);
                }
                return true;
            }
            else
            {
                return false;
            }
        }

        bool IEnumerator.MoveNext() => MoveNext();


        /// <summary>
        /// The current coordinate array is borrowed and reused across steps:
        /// copy it to retain coordinates.
        /// </summary>
        public int[] Current
        {
            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            get => Index;
        }

        object IEnumerator.Current
        {
            [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
            get => Index;
        }

        void IDisposable.Dispose() => Reset();
        public void Reset()
        {
            if (fixedDims is null || dims is null) return;
            iterator.Reset();
            Index = new int[length];
            fixedDims.CopyTo(Index, 0);
        }



    }
}
