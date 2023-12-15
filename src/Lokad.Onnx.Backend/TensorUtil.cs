using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public class TensorUtil
    {
        public static int HandleNegativeAxis(int axis, int tensorRank)
        {
            if (axis > tensorRank) throw new ArgumentException("T");
            return axis < 0 ? axis + tensorRank : axis; 
        }
    }
}
