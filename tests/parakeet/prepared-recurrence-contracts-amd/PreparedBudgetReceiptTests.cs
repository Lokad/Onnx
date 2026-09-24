namespace Lokad.Onnx.Backend.Tests;

public class PreparedBudgetReceiptTests
{
    [Fact]
    public void CompletePairIsPrepared()
    {
        // This anchor uses only pre-existing public types/members so the same
        // compiled test can establish the selected product's missing behavior.
        const int hidden = 640;
        const long pairBytes = 2L * 4 * hidden * hidden * sizeof(float);
        var graph = new ComputationalGraph(pairBytes);graph.Metadata["Name"] = "recurrent-budget-anchor";graph.Opset[""] = 17;
        graph.Inputs["x"] = Tensor<float>.Zeros(1, 1, hidden);
        graph.Initializers["w"] = Tensor<float>.Zeros(1, 4 * hidden, hidden);
        graph.Initializers["r"] = Tensor<float>.Zeros(1, 4 * hidden, hidden);
        graph.Outputs["y"] = Tensor<float>.Zeros(1, 1, 1, hidden);
        graph.Nodes.Add(new Node { Name = "lstm", Op = OpType.LSTM, Domain = "", OpTypeName = "LSTM", OpsetVersion = 17,
            Inputs = new[] { "x", "w", "r" }, Outputs = new[] { "y" }, Attributes = new() { ["hidden_size"] = hidden } });
        graph.Prepare();Assert.Equal(pairBytes, graph.RetainedPackedWeightBytes);
    }
}
