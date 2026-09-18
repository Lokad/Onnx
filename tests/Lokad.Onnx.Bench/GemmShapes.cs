using System.Collections.Generic;

namespace Lokad.Onnx.Bench;

// Canonical W3 shape matrix. PreparedB=true reproduces constant-weight dispatch
// (encoder projections, joint, linears); false reproduces dynamic products
// (attention). Neighbor/tail rows pin down dispatch cliffs.
public static class GemmShapes
{
    public static readonly IReadOnlyList<GemmShape> Canonical = new GemmShape[]
    {
        new GemmShape("enc-proj", 16, 1024, 1024, true, "encoder qkv/out projections"),
        new GemmShape("enc-ffn1", 16, 1024, 4096, true, "encoder feedforward up"),
        new GemmShape("enc-ffn2", 16, 4096, 1024, true, "encoder feedforward down"),
        new GemmShape("enc-pos", 31, 1024, 1024, true, "encoder linear_pos"),
        new GemmShape("enc-pre", 16, 4096, 1024, true, "encoder pre_encode out"),
        new GemmShape("att-scores", 16, 128, 16, false, "encoder attention QKt"),
        new GemmShape("att-weight", 16, 16, 128, false, "encoder attention AxV"),
        new GemmShape("att-out", 16, 128, 128, false, "encoder attention out"),
        new GemmShape("conf-pw1", 1024, 1024, 16, false, "conformer pointwise inner"),
        new GemmShape("conf-pw2", 2048, 1024, 16, false, "conformer pointwise inner"),
        new GemmShape("dec-joint", 40, 640, 8198, true, "decoder joint 8x5 folded"),
        new GemmShape("dec-pred", 5, 640, 640, true, "decoder pred"),
        new GemmShape("dec-enc", 8, 1024, 640, true, "decoder enc"),
        new GemmShape("seg-lin0", 589, 256, 128, true, "segmentation linear"),
        new GemmShape("seg-lin1", 589, 128, 128, true, "segmentation linear"),
        new GemmShape("seg-cls", 589, 128, 7, true, "segmentation classifier"),
        new GemmShape("tail-m15", 15, 1024, 1024, true, "row neighbor below 16"),
        new GemmShape("tail-m17", 17, 1024, 1024, true, "row neighbor above 16"),
        new GemmShape("tail-odd7", 7, 64, 64, false, "odd rows with fixup"),
        new GemmShape("tail-odd9", 9, 64, 64, false, "odd rows p65-adjacent"),
        new GemmShape("tail-m1", 1, 32, 16, false, "single-row lane"),
        new GemmShape("tail-tiny", 2, 2, 2, false, "minimal"),
        new GemmShape("tail-m100", 100, 256, 256, false, "non-64-multiple transient"),
        new GemmShape("tail-k256", 64, 256, 256, false, "256 reduction neighbor"),
        new GemmShape("tail-tiled", 80, 256, 1152, false, "composer two-sweep"),
    };
}
