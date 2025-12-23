namespace Lokad.Onnx;

public static class HardwareConfig
{
    public static bool UseSimd { get; set; } = true;

    public static bool UseIntrinsics { get; set; } = HardwareIntrinsics.IsX86FmaSupported;

    public static void EnableIntrinsics()
    {
        UseSimd = true;
        UseIntrinsics = true;
    }

    public static void EnableSimdOnly()
    {
        UseSimd = true;
        UseIntrinsics = false;
    }
}

