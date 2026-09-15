namespace Lokad.Onnx.Backend.Tests;
using Lokad.Onnx.Tests.Support;

// Pyannote embedding backbone conformance: fbank features [batch,frames,80]
// through the ResNet encoder to frame features [batch,2560,reducedFrames].
// The first facts use a deterministic synthetic input pinned by output mean
// and spot values, so they run offline with only the model file present.
// The last fact replays the real fbank fixture when it is present and
// skips otherwise.
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

        var fb = NpySupport.ReadFloat32(npyPath!);
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

    [SkippableFact]
    public void VoiceEmbedding_Replay400_Agrees()
    {
        var graph = ModelFixture.LoadRequiredModel("VoiceEmbedding", "models", "speaker-diarization-community-1", "onnx", "embedding", "embedding_encoder.onnx");
        var npyPath = ModelFixture.FindModelPath("models", "voice-fixtures", "replay", "embedding_fbank400.npy");
        Skip.If(npyPath is null, "Replay 400-frame fbank not present.");

        var fb = NpySupport.ReadFloat32(npyPath!);
        Assert.Equal(new[] { 1, 400, 80 }, fb.Shape);
        var input = new DenseTensor<float>(fb.Values, new[] { 1, 400, 80 });

        var feeds = new Dictionary<string, ITensor>();
        feeds[InputName] = input;
        ModelFixture.AssertExecuted(graph, graph.Execute(feeds, true));
        var output = (Tensor<float>)graph.Outputs[OutputName];
        Assert.Equal(new[] { 1, 2560, 50 }, output.Dimensions.ToArray());
        var values = ModelFixture.CheckedOutput(graph, OutputName);
        ModelFixture.AssertMean(values, 0.046228393912, 1e-6, "voice-embedding-400");
        ModelFixture.AssertSpots(values,
            new int[] { 1994, 3988, 6979, 22931, 25922, 41874 },
            new float[] { 0.33314839005470276f, 0.1358853578567505f, 0.6166670322418213f, 0.02999110519886017f, 1.9192506074905396f, 0.33672693371772766f },
            1e-3, "voice-embedding-400");
    }}
