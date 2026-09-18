using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace Lokad.Onnx;

public partial class MathOps
{
    // Same nonpositive-domain polynomial as ExpVectorNonpositive, at sixteen lanes.
    // Callers preserve the original eight-lane sum order after splitting the result.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Vector512<float> ExpVectorNonpositive512(Vector512<float> v)
    {
        var isNotNaN = Vector512.Equals(v, v);
        var underflow = Vector512.LessThan(v, Vector512.Create(-88.722839f));
        var x = Vector512.ConditionalSelect(underflow, Vector512<float>.Zero, v);
        var scaled = x * Vector512.Create(1.44269504088896341f);
        // Both +/-0.5 truncate to zero, including when the input is signed zero.
        var shifted = scaled - Vector512.Create(0.5f);
        var n = Vector512.ConvertToInt32(shifted);
        var clamped = Vector512.Max(n, Vector512.Create(-126));
        var nf = Vector512.ConvertToSingle(clamped);
        var r = Vector512.FusedMultiplyAdd(nf, Vector512.Create(-0.693359375f), x);
        r = Vector512.FusedMultiplyAdd(nf, Vector512.Create(2.12194440e-4f), r);
        var y = r * r;
        var tHi = Vector512.FusedMultiplyAdd(Vector512.Create(1f / 5040f), r, Vector512.Create(1f / 720f));
        var tLo = Vector512.FusedMultiplyAdd(Vector512.Create(1f / 6f), r, Vector512.Create(1f / 2f));
        var tMid = Vector512.FusedMultiplyAdd(Vector512.Create(1f / 120f), r, Vector512.Create(1f / 24f));
        var tOne = Vector512.FusedMultiplyAdd(Vector512.Create(1f), r, Vector512.Create(1f));
        var uHi = Vector512.FusedMultiplyAdd(tHi, y, tMid);
        var uLo = Vector512.FusedMultiplyAdd(tLo, y, tOne);
        var y2 = y * y;
        var p = Vector512.FusedMultiplyAdd(uHi, y2, uLo);
        var bits = Vector512.ShiftLeft(clamped + Vector512.Create(127), 23);
        var scale = Unsafe.As<Vector512<int>, Vector512<float>>(ref bits);
        var yy = p * scale;
        yy = Vector512.ConditionalSelect(underflow, Vector512<float>.Zero, yy);
        return Vector512.ConditionalSelect(isNotNaN, yy, Vector512.Create(float.NaN));
    }
}
