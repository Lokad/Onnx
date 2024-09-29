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
        public TensorDimensionsIterator(int[] dims)
        {
            if (dims is null) throw new ArgumentNullException("Can't construct TensorDimensionsIterator with an empty shape.");

            if (dims.Length == 0)
                dims = new int[] { 1 };

            dimensions = dims;
            Index = new int[dims.Length];
            resetto = subcursor = dimensions.Length - 1;
            endCallback = null;
        }

       
        public TensorDimensionsIterator(int[] dims, EndCallbackHandler endCallback) : this(dims)
        {
            this.endCallback = endCallback;
        }
        #endregion

        public IEnumerator<int[]> GetEnumerator() => this;

        IEnumerator IEnumerable.GetEnumerator() => this;

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        bool IEnumerator.MoveNext()
        {
            if (!moveStart)
            {
                moveStart = true;
                return Index != null;
            }
            else
            {
                return Next() != null;
            }
        }
        
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
        public int[] Next()
        {
            
            if (subcursor <= -1)
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
        private readonly EndCallbackHandler endCallback;
        private readonly int[] dimensions;
        private readonly int resetto;
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
        public TensorDimensionsIterator iterator = new TensorDimensionsIterator();
        public IEnumerator<int[]> iteratorEnumerator;
        public int[] Index = null;
        public int[] VariableIndex => iterator.Index;

        public TensorFixedDimensionsIterator(int[] fixedDimensions, params int[] dims)
        {
            this.fixedDims = fixedDimensions;
            this.dims = dims;
            this.length = fixedDimensions.Length + dims.Length;
            iterator = new TensorDimensionsIterator(dims);
            iteratorEnumerator = iterator.GetEnumerator();
            Index = new int[length];
            fixedDims.CopyTo(Index, 0);
        }

        public TensorFixedDimensionsIterator(ITensor t, Range r, params int[] dims) : this(t.Dims[r], dims)
        {

        }
        public IEnumerator<int[]> GetEnumerator() => this;

        IEnumerator IEnumerable.GetEnumerator() => this;

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        bool IEnumerator.MoveNext()
        { 
            if (iteratorEnumerator.MoveNext())
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
            iterator.Reset();
            Index = null;
        }



    }

}
