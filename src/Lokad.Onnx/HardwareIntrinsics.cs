namespace Lokad.Onnx;

using System.Collections.Generic;

using System.Numerics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using System.Text;


public static class HardwareIntrinsics
{
    public static string GetVectorSize() => Vector.IsHardwareAccelerated ? $"VectorSize={Vector<byte>.Count * 8}" : string.Empty;

    public static string GetShortInfo()
    {
        if (IsX86Avx512FSupported)
            return GetShortAvx512Representation();
        if (IsX86Avx2Supported)
            return "AVX2";
        else if (IsX86AvxSupported)
            return "AVX";
        else if (IsX86Sse42Supported)
            return "SSE4.2";
        else if (IsX86Sse41Supported)
            return "SSE4.1";
        else if (IsX86Ssse3Supported)
            return "SSSE3";
        else if (IsX86Sse3Supported)
            return "SSE3";
        else if (IsX86Sse2Supported)
            return "SSE2";
        else if (IsX86SseSupported)
            return "SSE";
        else if (IsX86BaseSupported)
            return "X86Base";
        else if (IsArmAdvSimdSupported)
            return "AdvSIMD";
        else if (IsArmBaseSupported)
            return "ArmBase";
        else
            return GetVectorSize(); // Runtimes prior to .NET Core 3.0 (APIs did not exist so we print non-exact Vector info)
    }

    public static string GetFullInfo()
    {
        return string.Join(",", GetCurrentProcessInstructionSets());

        static IEnumerable<string> GetCurrentProcessInstructionSets()
        {
            if (IsX86Avx512FSupported) yield return GetShortAvx512Representation();
            else if (IsX86Avx2Supported) yield return "AVX2";
            else if (IsX86AvxSupported) yield return "AVX";
            else if (IsX86Sse42Supported) yield return "SSE4.2";
            else if (IsX86Sse41Supported) yield return "SSE4.1";
            else if (IsX86Ssse3Supported) yield return "SSSE3";
            else if (IsX86Sse3Supported) yield return "SSE3";
            else if (IsX86Sse2Supported) yield return "SSE2";
            else if (IsX86SseSupported) yield return "SSE";
            else if (IsX86BaseSupported) yield return "X86Base";

            if (IsX86AesSupported) yield return "AES";
            if (IsX86Bmi1Supported) yield return "BMI1";
            if (IsX86Bmi2Supported) yield return "BMI2";
            if (IsX86FmaSupported) yield return "FMA";
            if (IsX86LzcntSupported) yield return "LZCNT";
            if (IsX86PclmulqdqSupported) yield return "PCLMUL";
            if (IsX86PopcntSupported) yield return "POPCNT";
            if (IsX86AvxVnniSupported) yield return "AvxVnni";
            if (IsX86SerializeSupported) yield return "SERIALIZE";
                    // TODO: Add MOVBE when API is added.
        }
    }

    public static bool IsX86BaseSupported => X86Base.IsSupported;

    public static bool IsX86SseSupported => Sse.IsSupported;

    public static bool IsX86Sse2Supported => Sse2.IsSupported;

    public static bool IsX86Sse3Supported => Sse3.IsSupported;

    public static bool IsX86Ssse3Supported => Ssse3.IsSupported;

    public static bool IsX86Sse41Supported => Sse41.IsSupported;

    public static bool IsX86Sse42Supported => Sse42.IsSupported;

    public static bool IsX86AvxSupported => Avx.IsSupported;

    public static bool IsX86Avx2Supported => Avx2.IsSupported;

    public static bool IsX86Avx512FSupported => Avx512F.IsSupported;

    public static bool IsX86Avx512FVLSupported => Avx512F.VL.IsSupported;

    public static bool IsX86Avx512BWSupported => Avx512BW.IsSupported;

    public static bool IsX86Avx512CDSupported => Avx512CD.IsSupported;

    public static bool IsX86Avx512DQSupported => Avx512DQ.IsSupported;

    public static bool IsX86Avx512VbmiSupported => Avx512Vbmi.IsSupported;

    public static bool IsX86AesSupported => System.Runtime.Intrinsics.X86.Aes.IsSupported;

    public static bool IsX86Bmi1Supported => Bmi1.IsSupported;

    public static bool IsX86Bmi2Supported => Bmi2.IsSupported;

    public static bool IsX86FmaSupported => Fma.IsSupported;

    public static bool IsX86LzcntSupported => Lzcnt.IsSupported;

    public static bool IsX86PclmulqdqSupported => Pclmulqdq.IsSupported;

    public static bool IsX86PopcntSupported => Popcnt.IsSupported;

    public static bool IsX86AvxVnniSupported =>
#pragma warning disable CA2252 // This API requires opting into preview features
        AvxVnni.IsSupported;
#pragma warning restore CA2252 // This API requires opting into preview features

    public static bool IsX86SerializeSupported => X86Serialize.IsSupported;

    public static bool IsArmBaseSupported => ArmBase.IsSupported;

    public static bool IsArmAdvSimdSupported => AdvSimd.IsSupported;

    public static bool IsArmAesSupported => System.Runtime.Intrinsics.Arm.Aes.IsSupported;

    public static bool IsArmCrc32Supported => Crc32.IsSupported;

    public static bool IsArmDpSupported => Dp.IsSupported;

    public static bool IsArmRdmSupported => Rdm.IsSupported;

    public static bool IsArmSha1Supported => Sha1.IsSupported;

    public static bool IsArmSha256Supported => Sha256.IsSupported;


    private static string GetShortAvx512Representation()
    {
        StringBuilder avx512 = new("AVX-512F");
        if (IsX86Avx512CDSupported) avx512.Append("+CD");
        if (IsX86Avx512BWSupported) avx512.Append("+BW");
        if (IsX86Avx512DQSupported) avx512.Append("+DQ");
        if (IsX86Avx512FVLSupported) avx512.Append("+VL");
        if (IsX86Avx512VbmiSupported) avx512.Append("+VBMI");

        return avx512.ToString();
    }
}
