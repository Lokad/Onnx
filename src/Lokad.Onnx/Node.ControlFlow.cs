namespace Lokad.Onnx;

using static OpResult;

public partial struct Node
{
    static string[] BranchOutputs(ComputationalGraph branch) => branch.OutputDescs.Count == 0
        ? branch.Outputs.Keys.ToArray() : branch.OutputDescs.Select(d => d.Name).ToArray();

    static string? BranchSupportError(ComputationalGraph branch)
    {
        foreach (var node in branch.Nodes)
        {
            if (!CPUExecutionProvider.SupportsNode(node)) return "Unsupported branch node " + node.Name + ": " + node.DescribeOperator();
            if (node.Op == OpType.If)
            {
                if (node.Attributes is null || !node.Attributes.TryGetValue("then_branch", out var tv) || tv is not ComputationalGraph then
                    || !node.Attributes.TryGetValue("else_branch", out var ev) || ev is not ComputationalGraph otherwise)
                    return "Nested If " + node.Name + " requires executable then_branch and else_branch graphs.";
                if (BranchContractError(then, otherwise, node.Outputs.Length, node.ResolvedOpsetVersion(branch)) is { } contractError)
                    return "Nested If " + node.Name + ": " + contractError;
                continue; // BranchContractError recursively validates both branches.
            }
            if (node.Attributes is not null)
                foreach (var value in node.Attributes.Values)
                    if (value is ComputationalGraph nested && BranchSupportError(nested) is { } error) return error;
        }
        return null;
    }

    static string? BranchContractError(ComputationalGraph thenBranch, ComputationalGraph elseBranch, int outputCount, int version)
    {
        var thenOutputs = BranchOutputs(thenBranch);
        var elseOutputs = BranchOutputs(elseBranch);
        if (thenOutputs.Length != outputCount || elseOutputs.Length != outputCount)
            return "Both If branches must have the same output count as the owning node.";
        for (int i = 0; i < outputCount; i++)
        {
            var td = thenBranch.OutputDescs.FirstOrDefault(d => d.Name == thenOutputs[i]);
            var ed = elseBranch.OutputDescs.FirstOrDefault(d => d.Name == elseOutputs[i]);
            if (td is not null && ed is not null)
            {
                if (td.ElementType != ed.ElementType) return "If branch output types must agree.";
                // Descriptors currently omit sequence element types, so they
                // cannot establish the required agreement across branches.
                if (td.ElementType == TensorElementType.Sequence) return "If currently supports tensor outputs only.";
                if (version > 0 && version < 11 && !td.Dims.SequenceEqual(ed.Dims))
                    return "Before opset 11, If branch output shapes must agree.";
            }
        }
        return BranchSupportError(thenBranch) ?? BranchSupportError(elseBranch);
    }

    OpResult ExecuteIf(ComputationalGraph graph, ExecutionOptions options)
    {
        var condition = InputTensor(graph, 0);
        if (condition is null) return MissingInput(OpType.If, "cond");
        if (condition is not Tensor<bool> boolean || condition.Length != 1)
            return WrongInputType(OpType.If, "cond", "If requires exactly one boolean element.", condition);
        if (Attributes is null || !Attributes.TryGetValue("then_branch", out var thenValue) || thenValue is not ComputationalGraph thenBranch)
            return MissingAttribute(OpType.If, "then_branch", "An executable graph is required.");
        if (!Attributes.TryGetValue("else_branch", out var elseValue) || elseValue is not ComputationalGraph elseBranch)
            return MissingAttribute(OpType.If, "else_branch", "An executable graph is required.");
        // Analyze first to detect cyclic graph objects before recursive validation.
        var reads = GraphCaptures.NodeInputs(this);
        foreach (var name in reads) if (!string.IsNullOrEmpty(name)) _ = graph.GetInputTensor(name);
        if (BranchContractError(thenBranch, elseBranch, Outputs.Length, ResolvedOpsetVersion(graph)) is { } error)
            return Failure(OpType.If, error);
        bool takeThen = boolean.GetValue(0);
        var branch = takeThen ? thenBranch : elseBranch;
        string branchName = takeThen ? "then_branch" : "else_branch";
        var names = BranchOutputs(branch);
        var captures = new Dictionary<string, ITensor>(StringComparer.Ordinal);
        foreach (string name in GraphCaptures.FreeVariables(branch)) captures.Add(name, graph.GetInputTensor(name));
        var execution = branch.CreateExecution(options);
        execution.BindCaptures(captures);
        bool succeeded = false;
        try
        {
            succeeded = execution.Execute(new Dictionary<string, ITensor>(), true, ExecutionProvider.CPU, options);
            if (!succeeded) return Failure(OpType.If, "Branch " + branchName + " failed: " + execution.LastErrorMessage, execution.LastErrorCause);
            var results = new ITensor[names.Length];
            for (int i = 0; i < names.Length; i++) results[i] = execution.Outputs[names[i]];
            return Success(OpType.If, results);
        }
        finally { graph.RecordSubgraph(this, branchName, execution, succeeded); }
    }
}
