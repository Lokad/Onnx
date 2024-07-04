namespace Lokad.Onnx;

public static class HardwareConfig
{
    public static bool UseSimd { get; set; } = false;

    public static bool UseIntrinsics { get; set; } = true;  
}

