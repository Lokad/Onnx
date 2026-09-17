namespace Lokad.Onnx.Backend.Tests;

using Lokad.Onnx.Bench;

/// <summary>
/// Pins the P7 representative bench rows: staged shapes, dtypes and input
/// names, slice prefixes matching the full step-1 fixtures, and zero versus
/// recorded-carried single-step states. Timed validation against live ORT
/// happens in the bench itself under --rows representative.
/// </summary>
public class VoiceRepresentativeCasesTests
{
    static string RequireRoot()
    {
        var dir = new DirectoryInfo(Directory.GetCurrentDirectory());
        while (dir is not null)
        {
            if (Directory.Exists(Path.Combine(dir.FullName, "models", "voice-fixtures", "replay")))
                return dir.FullName;
            dir = dir.Parent;
        }
        if (Environment.GetEnvironmentVariable(ModelFixture.LaneVariable) == "1")
            Assert.Fail("representative replay assets requested via " + ModelFixture.LaneVariable + "=1 but not found.");
        Skip.If(true, "replay assets not present; set " + ModelFixture.LaneVariable + "=1 to require them.");
        throw new InvalidOperationException("unreachable");
    }

    static void AssertNamed(ITensor tensor, string name, int[] dims, TensorElementType dtype)
    {
        Assert.Equal(name, tensor.Name);
        Assert.Equal(dims, tensor.Dims.ToArray());
        Assert.Equal(dtype, tensor.ElementType);
    }

    [Fact]
    public void Encoder64_ShapesAndDtypes()
    {
        string root = RequireRoot();
        var feeds = VoiceModelCases.EncoderInputs64(root);
        AssertNamed(feeds["audio_signal"], "audio_signal", new[] { 1, 128, 64 }, TensorElementType.Float);
        AssertNamed(feeds["length"], "length", new[] { 1 }, TensorElementType.Int64);
    }

    [Fact]
    public void Encoder256_ShapesAndDtypes()
    {
        string root = RequireRoot();
        var feeds = VoiceModelCases.EncoderInputs256(root);
        AssertNamed(feeds["audio_signal"], "audio_signal", new[] { 1, 128, 256 }, TensorElementType.Float);
        AssertNamed(feeds["length"], "length", new[] { 1 }, TensorElementType.Int64);
    }

    [Fact]
    public void Embedding400_ShapesAndDtypes()
    {
        string root = RequireRoot();
        var feeds = VoiceModelCases.EmbeddingInputs400(root);
        AssertNamed(feeds["fbank_features"], "fbank_features", new[] { 1, 400, 80 }, TensorElementType.Float);
    }

    [Fact]
    public void Embedding800_ShapesAndDtypes()
    {
        string root = RequireRoot();
        var feeds = VoiceModelCases.EmbeddingInputs800(root);
        AssertNamed(feeds["fbank_features"], "fbank_features", new[] { 1, 800, 80 }, TensorElementType.Float);
    }

    [Fact]
    public void Segmentation1s_ShapesAndDtypes()
    {
        string root = RequireRoot();
        var feeds = VoiceModelCases.SegmentationSynth1sInputs(root);
        AssertNamed(feeds["waveform"], "waveform", new[] { 1, 1, 16000 }, TensorElementType.Float);
    }

    [Fact]
    public void DecoderSingle_ZeroStatesSliceFirstStep()
    {
        string root = RequireRoot();
        var full = VoiceModelCases.DecoderStep1Inputs(root);
        var single = VoiceModelCases.DecoderStepSingleInputs(root);
        Assert.Equal(new[] { "encoder_outputs", "input_states_1", "input_states_2", "target_length", "targets" },
            single.Keys.OrderBy(k => k).ToArray());
        var enc = (Tensor<float>)single["encoder_outputs"];
        AssertNamed(enc, "encoder_outputs", new[] { 1, 1024, 1 }, TensorElementType.Float);
        var fullEnc = ((Tensor<float>)full["encoder_outputs"]).ToArray();
        var want = new float[1024];
        for (int c = 0; c < 1024; c++) want[c] = fullEnc[c * 8];
        Assert.Equal(want, enc.ToArray());
        var tgt = (Tensor<int>)single["targets"];
        AssertNamed(tgt, "targets", new[] { 1, 1 }, TensorElementType.Int32);
        var fullTgt = ((Tensor<int>)full["targets"]).ToArray();
        Assert.Equal(new[] { fullTgt[0] }, tgt.ToArray());
        var tlen = (Tensor<int>)single["target_length"];
        AssertNamed(tlen, "target_length", new[] { 1 }, TensorElementType.Int32);
        Assert.Equal(new[] { 1 }, tlen.ToArray());
        foreach (var key in new[] { "input_states_1", "input_states_2" })
        {
            var st = (Tensor<float>)single[key];
            AssertNamed(st, key, new[] { 2, 1, 640 }, TensorElementType.Float);
            Assert.All(st.ToArray(), v => Assert.Equal(0f, v));
        }
    }

    [Fact]
    public void PrefixFrames_SelectsFirstFramesByStride()
    {
        // [1,3,8] holding 0..23: frame f of each channel sits at stride 8,
        // so frame 0 is [0,8,16], not the flat [0,1,2] (L1 regression).
        var src = new DenseTensor<float>(System.Linq.Enumerable.Range(0, 24).Select(i => (float)i).ToArray(), new[] { 1, 3, 8 });
        var one = VoiceModelCases.PrefixFrames(src, 1);
        Assert.Equal(new[] { 1, 3, 1 }, one.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f, 8f, 16f }, one.ToArray());
        var two = VoiceModelCases.PrefixFrames(src, 2);
        Assert.Equal(new[] { 1, 3, 2 }, two.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f, 1f, 8f, 9f, 16f, 17f }, two.ToArray());
    }

    [Fact]
    public void PrefixFrames_HandlesBatches()
    {
        // [2,2,4] holding 0..15: batch 1 starts at offset 8.
        var src = new DenseTensor<float>(System.Linq.Enumerable.Range(0, 16).Select(i => (float)i).ToArray(), new[] { 2, 2, 4 });
        var one = VoiceModelCases.PrefixFrames(src, 1);
        Assert.Equal(new[] { 2, 2, 1 }, one.Dimensions.ToArray());
        Assert.Equal(new float[] { 0f, 4f, 8f, 12f }, one.ToArray());
    }

    [Fact]
    public void PrefixFrames_RejectsInvalidCounts()
    {
        var src = new DenseTensor<float>(new float[24], new[] { 1, 3, 8 });
        Assert.Throws<ArgumentOutOfRangeException>(() => VoiceModelCases.PrefixFrames(src, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => VoiceModelCases.PrefixFrames(src, 9));
    }

    [Fact]
    public void PrefixTokens_SelectsFirstTokensByStride()
    {
        // [2,5] holding 0..9: batch 1 starts at offset 5.
        var src = new DenseTensor<int>(System.Linq.Enumerable.Range(0, 10).ToArray(), new[] { 2, 5 });
        var one = VoiceModelCases.PrefixTokens(src, 1);
        Assert.Equal(new[] { 2, 1 }, one.Dimensions.ToArray());
        Assert.Equal(new[] { 0, 5 }, one.ToArray());
        var two = VoiceModelCases.PrefixTokens(src, 2);
        Assert.Equal(new[] { 2, 2 }, two.Dimensions.ToArray());
        Assert.Equal(new[] { 0, 1, 5, 6 }, two.ToArray());
    }

    [Fact]
    public void PrefixTokens_RejectsInvalidCounts()
    {
        var src = new DenseTensor<int>(new int[10], new[] { 2, 5 });
        Assert.Throws<ArgumentOutOfRangeException>(() => VoiceModelCases.PrefixTokens(src, 0));
        Assert.Throws<ArgumentOutOfRangeException>(() => VoiceModelCases.PrefixTokens(src, 6));
    }

    [Fact]
    public void DecoderSingle_CarriedStatesAreRecorded()
    {
        string root = RequireRoot();
        var single = VoiceModelCases.DecoderStepSingleInputs(root);
        var carried = VoiceModelCases.DecoderStepSingleCarriedInputs(root);
        foreach (var key in new[] { "input_states_1", "input_states_2" })
        {
            var zero = ((Tensor<float>)single[key]).ToArray();
            var kept = (Tensor<float>)carried[key];
            AssertNamed(kept, key, new[] { 2, 1, 640 }, TensorElementType.Float);
            var arr = kept.ToArray();
            Assert.False(arr.SequenceEqual(zero), key + " carried states must differ from zero.");
            Assert.True(arr.Any(v => v != 0f), key + " carried states must be nonzero.");
        }
        Assert.Equal(((Tensor<float>)single["encoder_outputs"]).ToArray(), ((Tensor<float>)carried["encoder_outputs"]).ToArray());
        Assert.Equal(((Tensor<int>)single["targets"]).ToArray(), ((Tensor<int>)carried["targets"]).ToArray());
    }
}
