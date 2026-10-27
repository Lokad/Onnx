using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Lokad.Onnx.Backend.Tests;

public class LiveIndexCollectionTests
{
    public static IEnumerable<object[]> Cases()
    {
        foreach (int side in new[] { 128, 512 })
        foreach (bool sequence in new[] { false, true })
        foreach (int alias in new[] { 0, 1, 2 }) // Discarded, Identity output, reshaped output.
            yield return [side, sequence, alias];
    }

    [Theory]
    [MemberData(nameof(Cases))]
    public void DeadRootsCollectDuringExecutionAndLiveAliasesRemain(int side, bool sequence, int alias)
    {
        var model = new OnnxModel
        {
            Opset = { [""] = 18 }, Inputs = { Desc("x", 1, 1, 2, 2) }, Outputs = { Desc("out") },
            Initializers = { Float("scales", [4], [1, 1, side / 2f, side / 2f]), Float("capture", [], [0]), Float("observe", [], [0]) },
            Nodes = { Node("Mul", ["x", "x"], "warm"),
                Node("ReduceSum", ["warm"], "warm-sum", new() { ["keepdims"] = 0L }),
                Node("Resize", ["x", "", "scales"], "big", new() { ["mode"] = "nearest" }) }
        };
        string source = "big";
        if (sequence)
        {
            model.Initializers.Add(Long("split", [side / 2, side / 2]));
            model.Initializers.Add(new OnnxTensor { Name = "first", ElementType = TensorElementType.Int64, Dims = [], Data = new long[] { 0 } });
            model.Initializers.Add(new OnnxTensor { Name = "second", ElementType = TensorElementType.Int64, Dims = [], Data = new long[] { 1 } });
            model.Nodes.Add(Node("SplitToSequence", ["big", "split"], "pieces", new() { ["axis"] = 2L }));
            // Capture the unselected item while the sequence owns both roots.
            model.Nodes.Add(Node("Mul", ["warm-sum", "capture"], "captured"));
            if (alias != 0) model.Nodes.Add(Node("SequenceAt", ["pieces", "second"], alias == 1 ? "keep" : "kept-value"));
            model.Nodes.Add(Node("SequenceAt", ["pieces", "first"], "selected"));
            model.Initializers.Add(Float("zero-scalar", [], [0]));
            source = "selected";
        }
        if (alias != 0)
        {
            int count = side * side / (sequence ? 2 : 1);
            if (alias == 2)
            {
                model.Initializers.Add(Long("view-shape", [count]));
                model.Nodes.Add(Node("Reshape", [sequence ? "kept-value" : "big", "view-shape"], "keep"));
            }
            else if (!sequence) model.Nodes.Add(Node("Identity", ["big"], "keep"));
            model.Outputs.Add(alias == 2 ? Desc("keep", count) : Desc("keep", 1, 1, sequence ? side / 2 : side, side));
        }
        model.Nodes.Add(Node("Mul", [source, sequence ? "zero-scalar" : "capture"], "zero"));
        model.Nodes.Add(Node("ReduceSum", ["zero"], "total", new() { ["keepdims"] = 0L }));
        model.Nodes.Add(Node("Mul", ["total", "observe"], "out"));
        var context = Model.Load(model, false).CreateExecution(new ExecutionOptions(OptimizationMode.Memory, TensorExecutionOptions.Scalar), 0);
        var state = new Observation(context, sequence);
        context.Initializers["capture"] = new Tap(state.Capture);
        context.Initializers["observe"] = new Tap(state.Observe);
        var input = new DenseTensor<float>(new float[] { 1, 2, 3, 4 }, [1, 1, 2, 2]);
        Assert.True(context.Execute(new Dictionary<string, ITensor> { ["x"] = input }, true), context.LastErrorMessage);
        Assert.True(state.Captured);
        Assert.False(state.PoolOwned);
        Assert.True(state.BindingDropped);
        Assert.Equal(side * side / (sequence ? 2 : 1), state.Length);
        Assert.Equal(alias != 0, state.AliveDuringRun);
        Assert.Equal(alias == 0 ? -1 : 1, state.IndexCount);
        Assert.Equal(0f, ((Tensor<float>)context.Outputs["out"]).GetValue(0));
        Assert.Equal(new float[] { 1, 2, 3, 4 }, input.ToArray());
        if (alias != 0)
        {
            var values = ((Tensor<float>)context.Outputs["keep"]).ToArray();
            for (int i = 0; i < values.Length; i++)
                Assert.Equal(sequence ? 3f + (i % side >= side / 2 ? 1 : 0)
                    : 1f + (i / side >= side / 2 ? 2 : 0) + (i % side >= side / 2 ? 1 : 0), values[i]);
        }
        GC.KeepAlive(context);
    }

    static OnnxValueInfo Desc(string name, params int[] shape) => new() { Name = name, ElementType = TensorElementType.Float, Dims = shape };
    static OnnxTensor Float(string name, int[] shape, float[] values) => new() { Name = name, ElementType = TensorElementType.Float, Dims = shape, Data = values };
    static OnnxTensor Long(string name, long[] values) => new() { Name = name, ElementType = TensorElementType.Int64, Dims = [values.Length], Data = values };
    static OnnxNode Node(string op, string[] inputs, string output, Dictionary<string, object>? attributes = null) =>
        new() { Name = output, OpType = op, Inputs = inputs, Outputs = [output], Attributes = attributes ?? new() };

    sealed class Tap(Action observe) : DenseTensor<float>(Array.Empty<int>())
    {
        public override float GetValue(int index) { observe(); return base.GetValue(index); }
    }

    sealed class Observation(GraphExecution context, bool sequence)
    {
        WeakReference<float[]>? array;
        internal bool Captured, PoolOwned, BindingDropped, AliveDuringRun;
        internal int Length, IndexCount = -1;

        [MethodImpl(MethodImplOptions.NoInlining)]
        internal void Capture()
        {
            if (Captured) return;
            var value = sequence ? ((TensorSequence)context.IntermediateOutputs["pieces"]!).Items[1] : context.IntermediateOutputs["big"]!;
            Assert.True(MemoryMarshal.TryGetArray((ReadOnlyMemory<float>)((Tensor<float>)value).Storage, out var memory));
            array = new WeakReference<float[]>(memory.Array!);
            Length = memory.Array!.Length;
            PoolOwned = context.ActivePool!.IsOwned(memory.Array);
            Captured = true;
        }

        internal void Observe()
        {
            BindingDropped = context.IntermediateOutputs[sequence ? "pieces" : "big"] is null;
            GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect();
            (AliveDuringRun, IndexCount) = Status(context, array!);
        }

        [MethodImpl(MethodImplOptions.NoInlining)]
        static (bool Alive, int Count) Status(GraphExecution context, WeakReference<float[]> array)
        {
            if (!array.TryGetTarget(out var value)) return (false, -1);
            var index = (Dictionary<Array, int>)typeof(ComputationalGraph).GetField("liveArrayUsers", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(context)!;
            return (true, index.TryGetValue(value, out int count) ? count : -1);
        }
    }
}
