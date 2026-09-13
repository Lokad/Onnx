namespace Lokad.Onnx.Backend.Tests;

// First managed speech-model conformance case (PLAN.md Milestone 2).
// Covers the pyannote embedding backbone: fbank features [batch,frames,80]
// through the ResNet encoder to frame features [batch,2560,reducedFrames].
// The tracked regression uses a deterministic synthetic input so it runs
// offline with only the model file present; the ignored voice probe
// validated the same graph against ORT 1.23.2 on real fbank slices
// (200/998/2998 frames, maxScaled 2.3e-06/3.1e-06/5.0e-06, tol 1e-4,
// bit-identical reuse). The third fact replays the real fbank fixture
// when it is present and skips otherwise.
public class GraphExecutionVoiceEmbeddingTests
{
    const string InputName = "fbank_features";
    const string OutputName = "/resnet/pool/Reshape_output_0";

    static DenseTensor<float> SyntheticFbank(int frames)
    {
        int n = frames * 80;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(Math.Sin(i * 0.01) * 2.0);
        return new DenseTensor<float>(data, new[] { 1, frames, 80 });
    }

    static (double Sum, float First, float Last) Fingerprint(float[] values)
    {
        double sum = 0;
        foreach (var v in values) sum += v;
        return (sum, values[0], values[values.Length - 1]);
    }

    [SkippableFact]
    public void CanInferWithVoiceEmbeddingBackbone()
    {
        var graph = ModelFixture.LoadRequiredModel("VoiceEmbedding", "models", "speaker-diarization-community-1", "onnx", "embedding", "embedding_encoder.onnx");

        var input = SyntheticFbank(200);
        var before = Fingerprint(input.ToArray());

        var feeds = new Dictionary<string, ITensor>();
        feeds[InputName] = input;
        ModelFixture.AssertExecuted(graph, graph.Execute(feeds, true));
        var output = (Tensor<float>)graph.Outputs[OutputName];
        Assert.Equal(new[] { 1, 2560, 25 }, output.Dimensions.ToArray());

        var values = ModelFixture.CheckedOutput(graph, OutputName);
        ModelFixture.AssertMean(values, 0.00363077456, 1e-6, "voice-embedding");
        ModelFixture.AssertSpots(values,
            new int[] { 40000, 22611, 49207 },
            new float[] { 0.100256085f, 0.143690437f, 0.00357984006f },
            1e-3, "voice-embedding");

        var after = Fingerprint(input.ToArray());
        Assert.Equal(before, after);

        var input2 = SyntheticFbank(200);
        var feeds2 = new Dictionary<string, ITensor>();
        feeds2[InputName] = input2;
        ModelFixture.AssertExecuted(graph, graph.Execute(feeds2, true));
        var again = ModelFixture.CheckedOutput(graph, OutputName);
        Assert.Equal(values.Length, again.Length);
        for (int i = 0; i < values.Length; i++) Assert.Equal(values[i], again[i]);
    }

    [SkippableFact]
    public void VoiceEmbedding_VaryingLengths_Agree()
    {
        var graph = ModelFixture.LoadRequiredModel("VoiceEmbedding", "models", "speaker-diarization-community-1", "onnx", "embedding", "embedding_encoder.onnx");

        var input = SyntheticFbank(120);
        var feeds = new Dictionary<string, ITensor>();
        feeds[InputName] = input;
        ModelFixture.AssertExecuted(graph, graph.Execute(feeds, true));
        var output = (Tensor<float>)graph.Outputs[OutputName];
        Assert.Equal(new[] { 1, 2560, 15 }, output.Dimensions.ToArray());
        var values = ModelFixture.CheckedOutput(graph, OutputName);
        ModelFixture.AssertMean(values, 0.00334302313, 1e-6, "voice-embedding-120");
    }

    [SkippableFact]
    public void VoiceEmbedding_RealFbankSlice_Agrees()
    {
        var graph = ModelFixture.LoadRequiredModel("VoiceEmbedding", "models", "speaker-diarization-community-1", "onnx", "embedding", "embedding_encoder.onnx");
        var npyPath = ModelFixture.FindModelPath("models", "voice-fixtures", "reference", "wespeaker-fbank", "fbank.npy");
        Skip.If(npyPath is null, "Real fbank fixture not present; synthetic facts above still cover the backbone.");

        var fb = ReadNpyFloat32(npyPath!);
        Assert.Equal(new[] { 2998, 80 }, fb.Shape);
        int useFrames = 200;
        var data = new float[useFrames * 80];
        Array.Copy(fb.Values, 0, data, 0, data.Length);
        var input = new DenseTensor<float>(data, new[] { 1, useFrames, 80 });

        var feeds = new Dictionary<string, ITensor>();
        feeds[InputName] = input;
        ModelFixture.AssertExecuted(graph, graph.Execute(feeds, true));
        var output = (Tensor<float>)graph.Outputs[OutputName];
        Assert.Equal(new[] { 1, 2560, 25 }, output.Dimensions.ToArray());
        var values = ModelFixture.CheckedOutput(graph, OutputName);
        ModelFixture.AssertMean(values, 0.00524577778, 1e-6, "voice-embedding-fbank");
        ModelFixture.AssertSpots(values,
            new int[] { 40000, 22611, 49207 },
            new float[] { 0f, 0f, 0.0640186667f },
            1e-3, "voice-embedding-fbank");
    }

    static (float[] Values, int[] Shape) ReadNpyFloat32(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        Span<byte> magic = stackalloc byte[6];
        fs.ReadExactly(magic);
        if (magic[0] != 0x93 || magic[1] != 78) throw new InvalidDataException("Bad NPY magic.");
        fs.ReadByte(); fs.ReadByte();
        Span<byte> lenBytes = stackalloc byte[2];
        fs.ReadExactly(lenBytes);
        int headerLen = lenBytes[0] | (lenBytes[1] << 8);
        var headerBytes = new byte[headerLen];
        fs.ReadExactly(headerBytes);
        string header = System.Text.Encoding.ASCII.GetString(headerBytes);
        if (!header.Contains("<f4")) throw new InvalidDataException("Expected float32 NPY.");
        int p0 = header.IndexOf('(');
        int p1 = header.IndexOf(')');
        var parts = header.Substring(p0 + 1, p1 - p0 - 1).Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        int[] shape = parts.Select(p => int.Parse(p)).ToArray();
        int n = 1;
        foreach (var d in shape) n *= d;
        var raw = new byte[n * 4];
        fs.ReadExactly(raw);
        var values = new float[n];
        Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
        return (values, shape);
    }
}
