namespace Lokad.Onnx.Backend.Tests;

// Parakeet encoder conformance (PLAN.md Milestone 3). Executes the 24-layer
// conformer encoder (opset 17, 4491 nodes, 2.4 GB external weights) through
// managed Lokad.Onnx: mel-scale features plus a frame count produce encoded
// frames and an exact integer encoded length. Expected values are frozen
// Python onnxruntime 1.29.0 oracles (ORT_SEQUENTIAL, intra/inter-op 1,
// ORT_ENABLE_ALL); the generator is the ignored
// .agent/voice-probe/ort129_encoder.py. Outputs stay near unit scale, so
// the suite-standard gates apply (mean 1e-6, spots 1e-3) with exact
// lengths; the ignored probe cross-checked C# ORT 1.23.2 at maxScaled
// 4.8e-07 with bit-identical repeated execution.
public class GraphExecutionEncoderTests
{
    static DenseTensor<float> MelFrames(int frames)
    {
        int n = 128 * frames;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(Math.Sin(i * 0.011) * 0.5);
        return new DenseTensor<float>(data, new[] { 1, 128, frames });
    }

    static Dictionary<string, ITensor> StepInputs(int frames)
    {
        return new Dictionary<string, ITensor>
        {
            ["audio_signal"] = MelFrames(frames),
            ["length"] = new DenseTensor<long>(new long[] { frames }, new[] { 1 }),
        };
    }

    static (double Sum, float First, float Last) Fingerprint(float[] values)
    {
        double sum = 0;
        foreach (var v in values) sum += v;
        return (sum, values[0], values[values.Length - 1]);
    }

    [SkippableFact]
    public void CanInferWithParakeetEncoder()
    {
        var graph = ModelFixture.LoadRequiredModel("ParakeetEncoder", "models", "parakeet-tdt-0.6b-v3", "onnx", "encoder-model.onnx");

        var feeds = StepInputs(128);
        var before = Fingerprint(((Tensor<float>)feeds["audio_signal"]).ToArray());

        ModelFixture.AssertExecuted(graph, graph.Execute(feeds, true));
        var encoded = (Tensor<float>)graph.Outputs["outputs"];
        Assert.Equal(new[] { 1, 1024, 16 }, encoded.Dimensions.ToArray());

        var values = ModelFixture.CheckedOutput(graph, "outputs");
        ModelFixture.AssertMean(values, 0.00010823947377502918, 1e-6, "encoder outputs");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 100, 1000, 5000, 10000, 8192, 16383 },
            new float[] { -0.0010888695251196623f, 0.004972534254193306f, 0.008661553263664246f, 0.020346835255622864f, 0.04964444786310196f, 0.03171032294631004f, 0.0027273856103420258f, 0.002207183977589011f, 0.004838298074901104f },
            1e-3, "encoder outputs");

        var lengths = (Tensor<long>)graph.Outputs["encoded_lengths"];
        Assert.Equal(new long[] { 16L }, lengths.ToArray());

        var after = Fingerprint(((Tensor<float>)feeds["audio_signal"]).ToArray());
        Assert.Equal(before, after);

        ModelFixture.AssertExecuted(graph, graph.Execute(StepInputs(128), true));
        var again = ModelFixture.CheckedOutput(graph, "outputs");
        Assert.Equal(values.Length, again.Length);
        for (int i = 0; i < values.Length; i++) Assert.Equal(values[i], again[i]);
    }

    [SkippableFact]
    public void EncoderShorterInput_MatchesOracle()
    {
        var graph = ModelFixture.LoadRequiredModel("ParakeetEncoder", "models", "parakeet-tdt-0.6b-v3", "onnx", "encoder-model.onnx");

        ModelFixture.AssertExecuted(graph, graph.Execute(StepInputs(64), true));
        var encoded = (Tensor<float>)graph.Outputs["outputs"];
        Assert.Equal(new[] { 1, 1024, 8 }, encoded.Dimensions.ToArray());

        var values = ModelFixture.CheckedOutput(graph, "outputs");
        ModelFixture.AssertMean(values, 1.002680801320821e-05, 1e-6, "encoder short outputs");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 100, 1000, 5000, 4096, 8191 },
            new float[] { 0.0002568874042481184f, 0.0040051937103271484f, -0.003697516629472375f, -0.009679900482296944f, -0.0048505705781280994f, -0.01961119845509529f, -0.0038403174839913845f, 0.0003224927932024002f },
            1e-3, "encoder short outputs");

        var lengths = (Tensor<long>)graph.Outputs["encoded_lengths"];
        Assert.Equal(new long[] { 8L }, lengths.ToArray());
    }
}
