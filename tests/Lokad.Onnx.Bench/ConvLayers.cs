using System.Collections.Generic;

namespace Lokad.Onnx.Bench;

// Canonical W4 layer list (canonical 200-frame geometry). ExpectBlocked marks v1
// admission; declined layers must still run the real legacy path.
public static class ConvLayers
{
    public static readonly IReadOnlyList<ConvLayer> Canonical = new ConvLayer[]
    {
        new ConvLayer("L1-32", 1, 32, 80, 200, 32, 3, 3, 1, 1, 1, true, true, true, "emb layer1 interior"),
        new ConvLayer("L2-64", 1, 64, 40, 100, 64, 3, 3, 1, 1, 1, true, true, true, "emb layer2"),
        new ConvLayer("L3-128", 1, 128, 20, 50, 128, 3, 3, 1, 1, 1, true, true, true, "emb layer3"),
        new ConvLayer("L4-256", 1, 256, 10, 25, 256, 3, 3, 1, 1, 1, true, true, true, "emb layer4 border-heavy"),
        new ConvLayer("S2-trans", 1, 32, 80, 200, 64, 3, 3, 2, 1, 1, true, true, false, "stride-2 transition"),
        new ConvLayer("SC-1x1", 1, 64, 40, 100, 128, 1, 1, 2, 1, 0, true, true, false, "1x1 shortcut"),
        new ConvLayer("NB-48", 1, 48, 20, 50, 48, 3, 3, 1, 1, 1, true, true, true, "48ch block-aligned control"),
        new ConvLayer("NB-40", 1, 40, 20, 50, 40, 3, 3, 1, 1, 1, true, true, false, "non-block-aligned"),
        new ConvLayer("stem", 1, 1, 80, 200, 32, 3, 3, 1, 1, 1, true, true, false, "single-channel control"),
    };
}
