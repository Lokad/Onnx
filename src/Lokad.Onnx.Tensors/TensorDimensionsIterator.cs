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

        public int[] Current => Index;

        [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
        public int[] Next()
        {
            unchecked
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
                            //TODO somehow can we skip all ones?
                            endCallback?.Invoke(ref this);
                            if (subcursor >= 0) //if callback has resetted it
                                return Index;
                            return null;
                        }
                    } while (dimensions[subcursor] <= 1);

                    ++Index[subcursor];
                    if (Index[subcursor] >= dimensions[subcursor])
                        goto _repeat;

                    subcursor = resetto;
                }
            }
            return Index;
        }

        public TensorDimensionsIterator Append(params int[] dims) => new TensorDimensionsIterator(dimensions.Concat(dims).ToArray());

        public SliceIndex[] AppendEllipsis() => Index.Select(i => new SliceIndex(i)).Append(SliceIndex.Ellipsis).ToArray();

        public SliceIndex[] PrependEllipsis() => Index.Select(i => new SliceIndex(i)).Prepend(SliceIndex.Ellipsis).ToArray();
        public SliceIndex[] AppendSliceIndices(params SliceIndex[] indices) => Index.Select(i => SliceIndex.FromObj(i)).Concat(indices).ToArray();  

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
}
