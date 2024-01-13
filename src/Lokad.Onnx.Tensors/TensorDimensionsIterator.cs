using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public struct TensorDimensionsIterator
    {
        public delegate void EndCallbackHandler(ref TensorDimensionsIterator incr);
        private readonly EndCallbackHandler endCallback;
        private readonly int[] dimensions;
        private readonly int resetto;
        public readonly int[] Index;
        private int subcursor;

   
        public TensorDimensionsIterator(int[] dims)
        {
            if (dims == null)
                throw new InvalidOperationException("Can't construct TensorDimensionsIterator with an empty shape.");

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

        public void Reset()
        {
            Array.Clear(Index, 0, Index.Length);
            subcursor = resetto;
        }

        [MethodImpl((MethodImplOptions)512)]
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

            return Index;
        }
    }
}
