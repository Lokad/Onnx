using System.Numerics;
using System.Runtime.CompilerServices;

namespace Lokad.Onnx;

public partial class MathOps
{
    // Frozen arithmetic twin of ErfVector. Keep the public reference unchanged
    // while measuring call/ABI costs in BiasGelu; tests compare raw lane bits.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector<float> ErfVectorInline(Vector<float> v)
    {
        var negZero = new Vector<float>(-0.0f);
        var signBits = Vector.BitwiseAnd(v, negZero);
        var ax = Vector.BitwiseAnd(Vector.OnesComplement(negZero), v);
        ax = Vector.ConditionalSelect(Vector.GreaterThan(ax, new Vector<float>(3.925f)), new Vector<float>(3.925f), ax);
        var sq = ax * ax;
        var rs = new Vector<float>(-5.99104969e-4f);
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(4.99339588e-3f));
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(-2.67667342e-2f));
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(1.12818025e-1f));
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(-3.76124859e-1f));
        rs = Vector.FusedMultiplyAdd(rs, sq, new Vector<float>(1.28379151e-1f));
        rs = Vector.FusedMultiplyAdd(rs, ax, ax);
        var big = Vector.GreaterThan(ax, new Vector<float>(0.921875f));
        rs = Vector.ConditionalSelect(big, Vector<float>.Zero, rs);
        var ab = Vector.ConditionalSelect(big, ax, Vector<float>.Zero);
        var rb = new Vector<float>(1.72948930e-5f);
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(-3.83208680e-4f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(3.88393435e-3f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(-2.42545605e-2f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(1.06777847e-1f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(6.34846687e-1f));
        rb = Vector.FusedMultiplyAdd(rb, ab, new Vector<float>(1.28717512e-1f));
        rb = Vector.FusedMultiplyAdd(rb, ab, ab);
        var t = Vector<float>.Zero - rb;
        t = Vector.ConditionalSelect(Vector.LessThan(t, new Vector<float>(-88.3762626647949f)), new Vector<float>(-88.3762626647949f), t);
        var r = Vector.FusedMultiplyAdd(new Vector<float>(1.44269504088896341f), t, new Vector<float>(12582912.0f));
        r = r - new Vector<float>(12582912.0f);
        var fx = Vector.FusedMultiplyAdd(r, new Vector<float>(-6.93145752e-1f), t);
        fx = Vector.FusedMultiplyAdd(r, new Vector<float>(-1.42860677e-6f), fx);
        var y = new Vector<float>(1.38319808e-3f);
        y = Vector.FusedMultiplyAdd(y, fx, new Vector<float>(8.37550033e-3f));
        y = Vector.FusedMultiplyAdd(y, fx, new Vector<float>(4.16689515e-2f));
        y = Vector.FusedMultiplyAdd(y, fx, new Vector<float>(1.66664466e-1f));
        y = Vector.FusedMultiplyAdd(y, fx, new Vector<float>(4.99999851e-1f));
        y = Vector.FusedMultiplyAdd(y, fx, Vector<float>.One);
        y = Vector.FusedMultiplyAdd(y, fx, Vector<float>.One);
        var ri = Vector.ConvertToInt32(r);
        ri = Vector.Min(Vector.Max(ri, new Vector<int>(-126)), new Vector<int>(127));
        y = y * Vector.AsVectorSingle(Vector.ShiftLeft(ri + new Vector<int>(127), 23));
        y = Vector<float>.One - y;
        y = Vector.BitwiseOr(rs, y);
        return Vector.BitwiseOr(y, signBits);
    }
}
