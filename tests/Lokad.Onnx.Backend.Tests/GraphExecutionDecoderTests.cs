namespace Lokad.Onnx.Backend.Tests;

// Parakeet decoder/joint conformance (PLAN.md Milestone 3). Executes the
// combined prediction/joint graph (opset 17) through managed Lokad.Onnx:
// encoder frames plus token/state inputs produce transducer scores,
// an exact integer length output, and updated recurrent states. Step two
// chains the step-one output states back as inputs, proving recurrent
// state carries across runs. Expected values are frozen Python
// onnxruntime 1.29.0 oracles (ORT_SEQUENTIAL, intra/inter-op 1,
// ORT_ENABLE_ALL); the generator is the ignored
// .agent/voice-probe/ort129_decoder.py. Logit magnitudes reach ~4000, so
// score tolerances are absolute (0.01 mean, 0.1 spots) at roughly the
// 1e-4 scaled gate used for engine comparisons; states stay near unit
// scale with tight gates, and lengths compare exactly.
public class GraphExecutionDecoderTests
{
    const int Frames = 8;
    const int Steps = 5;

    static DenseTensor<float> EncoderFrames()
    {
        int n = 1024 * Frames;
        var data = new float[n];
        for (int i = 0; i < n; i++) data[i] = (float)(Math.Sin(i * 0.013) * 1.5);
        return new DenseTensor<float>(data, new[] { 1, 1024, Frames });
    }

    static Dictionary<string, ITensor> StepInputs(float[] state1, float[] state2)
    {
        return new Dictionary<string, ITensor>
        {
            ["encoder_outputs"] = EncoderFrames(),
            ["targets"] = new DenseTensor<int>(new int[] { 1, 2, 3, 4, 5 }, new[] { 1, Steps }),
            ["target_length"] = new DenseTensor<int>(new int[] { Steps }, new[] { 1 }),
            ["input_states_1"] = new DenseTensor<float>(state1, new[] { 2, 1, 640 }),
            ["input_states_2"] = new DenseTensor<float>(state2, new[] { 2, 1, 640 }),
        };
    }

    static (double Sum, float First, float Last) Fingerprint(float[] values)
    {
        double sum = 0;
        foreach (var v in values) sum += v;
        return (sum, values[0], values[values.Length - 1]);
    }

    [SkippableFact]
    public void CanInferWithParakeetDecoderJoint()
    {
        var graph = ModelFixture.LoadRequiredModel("ParakeetDecoder", "models", "parakeet-tdt-0.6b-v3", "onnx", "decoder_joint-model.onnx");

        var feeds = StepInputs(new float[2 * 640], new float[2 * 640]);
        var before = Fingerprint(((Tensor<float>)feeds["encoder_outputs"]).ToArray());

        ModelFixture.AssertExecuted(graph, graph.Execute(feeds, true));
        var scores = (Tensor<float>)graph.Outputs["outputs"];
        Assert.Equal(new[] { 1, Frames, Steps, 8198 }, scores.Dimensions.ToArray());

        var values = ModelFixture.CheckedOutput(graph, "outputs");
        ModelFixture.AssertMean(values, -2019.813232421875, 0.01, "decoder scores");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 100, 1000, 8197, 8198, 16396, 163960, 327919 },
            new float[] { -1306.59521484375f, -1135.618896484375f, -1138.2774658203125f, -1135.1395263671875f, -2343.5908203125f, -221.34381103515625f, -1307.28369140625f, -1305.310302734375f, -1307.515625f, -225.1945037841797f },
            0.1, "decoder scores");

        var lengths = (Tensor<int>)graph.Outputs["prednet_lengths"];
        Assert.Equal(new int[] { 5 }, lengths.ToArray());

        var s1 = ModelFixture.CheckedOutput(graph, "output_states_1");
        ModelFixture.AssertMean(s1, 0.0024378797970712185, 1e-6, "decoder state 1");
        ModelFixture.AssertSpots(s1,
            new int[] { 0, 1, 639, 640, 1279 },
            new float[] { -0.737766683101654f, 0.20851925015449524f, -0.08223405480384827f, -0.00011066770093748346f, -0.0867960974574089f },
            1e-3, "decoder state 1");

        var s2 = ModelFixture.CheckedOutput(graph, "output_states_2");
        ModelFixture.AssertMean(s2, 0.004218598362058401, 1e-6, "decoder state 2");
        ModelFixture.AssertSpots(s2,
            new int[] { 0, 1, 639, 640, 1279 },
            new float[] { -0.9862733483314514f, 0.9998062252998352f, -0.08242437243461609f, -0.007138102315366268f, -0.38983362913131714f },
            1e-3, "decoder state 2");

        var after = Fingerprint(((Tensor<float>)feeds["encoder_outputs"]).ToArray());
        Assert.Equal(before, after);

        // Repeated execution with fresh inputs reproduces every value.
        ModelFixture.AssertExecuted(graph, graph.Execute(StepInputs(new float[2 * 640], new float[2 * 640]), true));
        var again = ModelFixture.CheckedOutput(graph, "outputs");
        Assert.Equal(values.Length, again.Length);
        for (int i = 0; i < values.Length; i++) Assert.Equal(values[i], again[i]);
    }

    [SkippableFact]
    public void DecoderChainedStates_MatchOracle()
    {
        var graph = ModelFixture.LoadRequiredModel("ParakeetDecoder", "models", "parakeet-tdt-0.6b-v3", "onnx", "decoder_joint-model.onnx");

        ModelFixture.AssertExecuted(graph, graph.Execute(StepInputs(new float[2 * 640], new float[2 * 640]), true));
        var s1 = ((Tensor<float>)graph.Outputs["output_states_1"]).ToArray();
        var s2 = ((Tensor<float>)graph.Outputs["output_states_2"]).ToArray();

        graph.Reset();
        ModelFixture.AssertExecuted(graph, graph.Execute(StepInputs(s1, s2), true));
        var values = ModelFixture.CheckedOutput(graph, "outputs");
        ModelFixture.AssertMean(values, -2015.919921875, 0.01, "decoder chained scores");
        ModelFixture.AssertSpots(values,
            new int[] { 0, 1, 2, 100, 1000, 8197, 8198, 16396, 163960, 327919 },
            new float[] { -1298.8681640625f, -1133.08056640625f, -1135.67724609375f, -1132.55078125f, -2334.6201171875f, -224.3458251953125f, -1309.9771728515625f, -1303.155517578125f, -1299.308837890625f, -226.57601928710938f },
            0.1, "decoder chained scores");

        var chained1 = ModelFixture.CheckedOutput(graph, "output_states_1");
        ModelFixture.AssertMean(chained1, 0.001916864886879921, 1e-6, "decoder chained state 1");
        var chained2 = ModelFixture.CheckedOutput(graph, "output_states_2");
        ModelFixture.AssertMean(chained2, -0.03472035750746727, 1e-6, "decoder chained state 2");

        var lengths = (Tensor<int>)graph.Outputs["prednet_lengths"];
        Assert.Equal(new int[] { 5 }, lengths.ToArray());
    }
}
