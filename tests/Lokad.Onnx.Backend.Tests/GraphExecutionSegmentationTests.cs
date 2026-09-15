namespace Lokad.Onnx.Backend.Tests;

// Pyannote segmentation conformance (PLAN.md Milestone 3). Executes the
// full diarization frontend (opset 17, 144 nodes, one If with two imported
// branches, four bidirectional LSTMs) through managed Lokad.Onnx: a mono
// waveform produces per-frame seven-class scores. Expected values are
// frozen Python onnxruntime 1.29.0 oracles (ORT_SEQUENTIAL, intra/inter-op
// 1, ORT_ENABLE_ALL); generators live in the ignored .agent/voice-probe
// (ort129_segsyn.py, ort129_segreal.py). Score magnitudes reach ~10, so
// the gates are absolute (mean 1e-4, spots 1e-3): the 10-second operating
// point amplifies cross-implementation rounding through high-gain
// normalization channels and recurrence (see the commit message), while
// every kernel in isolation matches ORT to 4e-6 or better.
public class GraphExecutionSegmentationTests
{
    static DenseTensor<float> SyntheticWave(int samples)
    {
        var data = new float[samples];
        for (int i = 0; i < samples; i++)
            data[i] = (float)(Math.Sin(i * 0.05) * 0.3 + Math.Sin(i * 0.013) * 0.2);
        return new DenseTensor<float>(data, new[] { 1, 1, samples });
    }

    static (double Sum, float First, float Last) Fingerprint(float[] values)
    {
        double sum = 0;
        foreach (var v in values) sum += v;
        return (sum, values[0], values[values.Length - 1]);
    }

    [SkippableFact]
    public void CanInferWithPyannoteSegmentation()
    {
        var graph = ModelFixture.LoadRequiredModel("PyannoteSegmentation", "models", "speaker-diarization-community-1", "onnx", "segmentation", "model.onnx");

        var feeds = new Dictionary<string, ITensor> { ["waveform"] = SyntheticWave(16000) };
        var before = Fingerprint(((Tensor<float>)feeds["waveform"]).ToArray());

        ModelFixture.AssertExecuted(graph, graph.Execute(feeds, true));
        var scores = (Tensor<float>)graph.Outputs["scores"];
        Assert.Equal(new[] { 1, 56, 7 }, scores.Dimensions.ToArray());

        var values = ModelFixture.CheckedOutput(graph, "scores");
        ModelFixture.AssertMean(values, -2.8473634719848633, 1e-4, "segmentation scores");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 6, 100, 130, 196, 391 },
            new float[] { -0.713207483291626f, -2.6290931701660156f, -3.6205379962921143f, -2.033146858215332f, -4.699270725250244f, -0.5564402341842651f, -3.932426929473877f },
            1e-3, "segmentation scores");

        var after = Fingerprint(((Tensor<float>)feeds["waveform"]).ToArray());
        Assert.Equal(before, after);

        ModelFixture.AssertExecuted(graph, graph.Execute(new Dictionary<string, ITensor> { ["waveform"] = SyntheticWave(16000) }, true));
        var again = ModelFixture.CheckedOutput(graph, "scores");
        Assert.Equal(values.Length, again.Length);
        for (int i = 0; i < values.Length; i++) Assert.Equal(values[i], again[i]);
    }

    [SkippableFact]
    public void SegmentationLongWindow_MatchesOracle()
    {
        var graph = ModelFixture.LoadRequiredModel("PyannoteSegmentation", "models", "speaker-diarization-community-1", "onnx", "segmentation", "model.onnx");

        ModelFixture.AssertExecuted(graph, graph.Execute(new Dictionary<string, ITensor> { ["waveform"] = SyntheticWave(160000) }, true));
        var scores = (Tensor<float>)graph.Outputs["scores"];
        Assert.Equal(new[] { 1, 589, 7 }, scores.Dimensions.ToArray());

        var values = ModelFixture.CheckedOutput(graph, "scores");
        ModelFixture.AssertMean(values, -4.790576934814453, 1e-4, "segmentation long scores");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 6, 100, 1374, 2061, 4122 },
            new float[] { -0.34140560030937195f, -4.082369804382324f, -4.441305160522461f, -3.0401368141174316f, -4.491325378417969f, -3.3029565811157227f, -5.285151958465576f },
            1e-3, "segmentation long scores");
    }

    [SkippableFact]
    public void SegmentationRealFixture_MatchesOracle()
    {
        var graph = ModelFixture.LoadRequiredModel("PyannoteSegmentation", "models", "speaker-diarization-community-1", "onnx", "segmentation", "model.onnx");
        var wavPath = ModelFixture.FindModelPath("models", "voice-fixtures", "audio", "pyannote-sample.wav");
        Skip.If(wavPath is null, "Real audio fixture not present; synthetic facts above still cover the graph.");

        var pcm = ReadMono16kwav(wavPath!);
        Assert.True(pcm.Length >= 160000, "Fixture holds a full 10-second window.");
        var window = new float[160000];
        Array.Copy(pcm, 0, window, 0, window.Length);
        var feeds = new Dictionary<string, ITensor> { ["waveform"] = new DenseTensor<float>(window, new[] { 1, 1, 160000 }) };

        ModelFixture.AssertExecuted(graph, graph.Execute(feeds, true));
        var scores = (Tensor<float>)graph.Outputs["scores"];
        Assert.Equal(new[] { 1, 589, 7 }, scores.Dimensions.ToArray());

        var values = ModelFixture.CheckedOutput(graph, "scores");
        ModelFixture.AssertMean(values, -5.001010894775391, 1e-4, "segmentation fixture scores");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 6, 100, 1000, 2000, 3000, 4122 },
            new float[] { -0.2283589243888855f, -4.351859092712402f, -5.141579627990723f, -3.1318717002868652f, -6.9208760261535645f, -7.315133571624756f, -4.658246040344238f, -2.5767836570739746f },
            1e-3, "segmentation fixture scores");
    }

    [SkippableFact]
    public void SegmentationIf_CarriesBothBranches()
    {
        // Tripwire against first-fixture specialization: the imported If
        // must carry both executable branch graphs (single-Conv then path,
        // 11-node chunked else path after 17 literal plus 6 computed branch
        // folds) with their outer captures available.
        var graph = ModelFixture.LoadRequiredModel("PyannoteSegmentation", "models", "speaker-diarization-community-1", "onnx", "segmentation", "model.onnx");
        var node = graph.Nodes.First(n => n.Op == OpType.If);
        Assert.True(node.Attributes!.TryGetValue("then_branch", out var then) && then is ComputationalGraph);
        Assert.True(node.Attributes!.TryGetValue("else_branch", out var els) && els is ComputationalGraph);
        Assert.Equal(1, ((ComputationalGraph)then).Nodes.Count);
        Assert.Equal(11, ((ComputationalGraph)els).Nodes.Count);
    }

    // Minimal test-local RIFF reader for 16 kHz mono PCM16 fixtures.
    // Kept inside the test so the shipped library gains no audio format.
    static float[] ReadMono16kwav(string path)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);
        using var br = new BinaryReader(fs);
        if (new string(br.ReadChars(4)) != "RIFF") throw new InvalidDataException("Not a RIFF file.");
        br.ReadInt32();
        if (new string(br.ReadChars(4)) != "WAVE") throw new InvalidDataException("Not a WAVE file.");
        byte[]? pcm = null;
        while (fs.Position < fs.Length)
        {
            string chunk = new string(br.ReadChars(4));
            int size = br.ReadInt32();
            if (chunk == "fmt ")
            {
                var fmt = br.ReadBytes(16);
                if (BitConverter.ToUInt16(fmt, 0) != 1 || BitConverter.ToUInt16(fmt, 2) != 1
                    || BitConverter.ToInt32(fmt, 4) != 16000 || BitConverter.ToUInt16(fmt, 14) != 16)
                    throw new InvalidDataException("Expected 16 kHz mono PCM16.");
                if (size > 16) br.ReadBytes(size - 16);
            }
            else if (chunk == "data")
            {
                pcm = br.ReadBytes(size);
            }
            else br.ReadBytes(size);
        }
        if (pcm is null) throw new InvalidDataException("No data chunk.");
        var samples = new float[pcm.Length / 2];
        for (int i = 0; i < samples.Length; i++)
            samples[i] = BitConverter.ToInt16(pcm, 2 * i) / 32768f;
        return samples;
    }
}
