using System.Text.Json;
using Lokad.Onnx;

static partial class Program
{
    static Tensor<T> Invoke<T>(Tensor<bool> condition, Tensor<T> x, Tensor<T> y) where T : unmanaged
    {
        LastValidationStages = 0;
        LastStages = [];
        if (Boundary == "tensor") return Tensor<T>.Where(condition, x, y);
        if (Mode == "codegen")
        {
            LastValidationStages = -1;
            var result = CPUExecutionProvider.Where(condition, x, y, null);
            Require(result.Op == OpType.Where && result.Status == OpStatus.Success && result.Outputs.Length == 1
                && result.Inputs.Length == 0 && result.Message is null && result.Cause is null, "provider codegen result");
            return (Tensor<T>)result.Outputs[0];
        }
        using var context = Profiler.BeginExecution(true);
        context.StartNodeProfile(0, OpType.Where);
        try
        {
            var result = CPUExecutionProvider.Where(condition, x, y, null);
            Require(result.Op == OpType.Where && result.Status == OpStatus.Success && result.Outputs.Length == 1
                && result.Inputs.Length == 0 && result.Message is null && result.Cause is null, "provider result");
            return (Tensor<T>)result.Outputs[0];
        }
        finally
        {
            context.StopNodeProfile();
            Require(context.Profile.Count == 1, "provider profile node");
            LastStages = context.Profile.Peek().OpsProfile.Reverse().Select(p => (int)p.Stage).ToArray();
            LastValidationStages = LastStages.Count(p => p == (int)OpStage.ValidateArguments);
        }
    }

    static void CheckFailure(OpResult value, string? exactMessage = null)
    {
        Require(value.Op == OpType.Where && value.Status == OpStatus.Failure && value.Outputs.Length == 0
            && value.Inputs.Length == 0 && value.Cause is null && !string.IsNullOrWhiteSpace(value.Message), "provider failure result");
        if (exactMessage is not null) Require(value.Message == exactMessage, "provider failure message");
    }

    static object[] ProviderContracts()
    {
        var rows = new List<object>(); var held = new List<Tensor<float>>();
        var cs = new bool[4096]; var xs = Values<float>(1, 4); var ys = Values<float>(4096, 0);
        var c = new DenseTensor<bool>(cs.AsMemory(), new[] { 4096 });
        var x = new DenseTensor<float>(xs.AsMemory(), Array.Empty<int>());
        var y = new DenseTensor<float>(ys.AsMemory(), new[] { 4096 });
        string[] Stores() => [Hash(Bytes(cs)), Hash(Bytes(xs)), Hash(Bytes(ys))];
        var before = Stores(); var expected = Bytes(ys); var expectedHash = Hash(expected);
        var badDegree = new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(false, false, 0));
        var badIntrinsics = new ExecutionOptions(OptimizationMode.Speed, new TensorExecutionOptions(false, true, 1));
        (string Name, ExecutionOptions? Value)[] options = [
            ("null", null), ("default", ExecutionOptions.Default), ("scalar", ExecutionOptions.Scalar),
            ("simd", ExecutionOptions.Simd), ("intrinsics", ExecutionOptions.Intrinsics), ("memory", ExecutionOptions.Memory),
            ("parallel2", new(OptimizationMode.Speed, TensorExecutionOptions.Parallel(2))),
            ("no-pool", new(OptimizationMode.Speed, TensorExecutionOptions.Auto with { DisableBufferPool = true }))];
        foreach (var option in options)
        {
            var result = CPUExecutionProvider.Where(c, x, y, option.Value);
            Require(result.Op == OpType.Where && result.Status == OpStatus.Success && result.Outputs.Length == 1
                && result.Inputs.Length == 0 && result.Message is null && result.Cause is null, "valid option result");
            var output = (Tensor<float>)result.Outputs[0];
            Require(output.Dimensions.SequenceEqual(y.Dimensions) && Bytes(Logical(output)).SequenceEqual(expected), "option output");
            Require(!ReferenceEquals(output, y) && held.All(h => !ReferenceEquals(h, output)), "option ownership");
            held.Add(output); Require(before.SequenceEqual(Stores()), "option input changed");
            rows.Add(new { name = "option-" + option.Name, output = expectedHash, values = 4096, inputs = true, owned = true });
        }
        foreach (var output in held) Require(Bytes(Logical(output)).SequenceEqual(expected), "held option output");
        var mutated = Logical(held[0]); Mutate(mutated); held[0].SetValue(0, mutated[0]);
        foreach (var output in held.Skip(1)) Require(Bytes(Logical(output)).SequenceEqual(expected), "option outputs alias");
        Require(before.SequenceEqual(Stores()), "option output aliases input");

        void Failure(string name, ITensor? condition, ITensor? left, ITensor? right, ExecutionOptions? opts = null, string? message = null)
        {
            var value = CPUExecutionProvider.Where(condition, left, right, opts); CheckFailure(value, message);
            Require(before.SequenceEqual(Stores()), name + " failure changed inputs");
            rows.Add(new { name, failure = value.Message, inputs = true });
        }
        Failure("missing-condition", null, x, y, message: "The required input parameter condition is missing or null.");
        Failure("missing-X", c, null, y, message: "The required input parameter X is missing or null.");
        Failure("missing-Y", c, x, null, message: "The required input parameter Y is missing or null.");
        var ints = DenseTensor<int>.OfValues(new[] { 1 });
        Failure("wrong-condition", ints, x, y);
        Failure("mismatched-operands", c, x, ints);
        foreach (var (name, operand) in new (string, ITensor)[] {
            ("int8", DenseTensor<sbyte>.OfValues(new sbyte[] { 1 })),
            ("int16", DenseTensor<short>.OfValues(new short[] { 1 })),
            ("uint16", DenseTensor<ushort>.OfValues(new ushort[] { 1 })) })
            Failure("unsupported-" + name, c, operand, operand);
        void Error(string name, ITensor condition, ExecutionOptions opts, Type expectedType, string expectedParameter)
        {
            Exception? error = null;
            try { CPUExecutionProvider.Where(condition, x, y, opts); } catch (Exception e) { error = e; }
            Require(error?.GetType() == expectedType && ((ArgumentException)error).ParamName == expectedParameter, name + " exception");
            Require(before.SequenceEqual(Stores()), name + " error changed inputs");
            rows.Add(new { name, error = error!.GetType().FullName, parameter = ((ArgumentException)error).ParamName, inputs = true });
        }
        Error("invalid-parallelism", c, badDegree, typeof(ArgumentOutOfRangeException), "MaxDegreeOfParallelism");
        Error("intrinsics-without-simd", c, badIntrinsics, typeof(ArgumentException), "UseIntrinsics");
        Failure("missing-before-options", null, x, y, badDegree, "The required input parameter condition is missing or null.");
        Error("options-before-condition-type", ints, badDegree, typeof(ArgumentOutOfRangeException), "MaxDegreeOfParallelism");
        Require(rows.Count == 20, "contract census"); return rows.ToArray();
    }
}
