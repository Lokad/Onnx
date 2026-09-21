using System.Runtime.InteropServices;
using System.Text.Json;
using Lokad.Onnx;

static class Capture
{
    public static void Run(string root, string manifestPath, string output)
    {
        var spec = Support.Read(manifestPath);
        foreach (var file in spec.GetProperty("models").EnumerateArray()) Support.PathOf(root, file);
        var graph = OnnxImport.Load(Support.PathOf(root, spec.GetProperty("encoder")), 256L*1024*1024)
            ?? throw new InvalidDataException(OnnxImport.LastErrorMessage);
        var nodes = new[] { 99, 102, 109 }.Select(id => graph.Nodes.Single(n => n.ID == id)).ToArray();
        Support.Require(nodes.All(n => n.Op == OpType.MatMul), "Selected nodes");
        var taps = nodes.SelectMany(n => new[] { n.Inputs[0], n.Outputs[0] }).Distinct().ToArray();
        foreach (string tap in taps)
        {
            graph.Outputs[tap] = null;
            graph.OutputDescs.Add(new OnnxValueInfo { Name = tap, ElementType = TensorElementType.Float, Dims = new[] { 1,-1,-1 } });
        }
        graph.InvalidatePreparation();
        var entries = new List<object>(); var requests = new List<object>(); var weights = new Dictionary<string, object>();
        var initializers = nodes.Select(n => n.Inputs[1]).Distinct().ToDictionary(n => n, n => Support.Hash(Support.Bytes(graph.Initializers[n])));
        object Save(string file, Tensor<float> tensor)
        {
            byte[] bytes = Support.Bytes(tensor); Support.Require(tensor.ToArray().All(float.IsFinite), "Nonfinite");
            using (var f = new FileStream(Path.Combine(output,file),FileMode.CreateNew)) f.Write(bytes);
            return new { file, shape = tensor.Dimensions.ToArray(), strides = tensor.Strides.ToArray(), reverse = tensor.IsReversedStride,
                type = tensor.GetType().Name, bytes = bytes.Length, sha256 = Support.Hash(bytes), dtype = "<f4" };
        }
        int index = 0;
        foreach (var c in spec.GetProperty("cases").EnumerateArray())
        {
            string name = c.GetProperty("name").GetString()!; int frames = c.GetProperty("frames").GetInt32();
            var feeds = new Dictionary<string, ITensor>(); var inputBits = new Dictionary<string,string>();
            foreach (var pair in c.GetProperty("inputs").EnumerateObject())
            {
                byte[] bytes = File.ReadAllBytes(Support.PathOf(root,pair.Value)); int[] shape = Support.Shape(pair.Value);
                ITensor t = pair.Value.GetProperty("dtype").GetString() == "<f4"
                    ? new DenseTensor<float>(MemoryMarshal.Cast<byte,float>(bytes).ToArray(),shape)
                    : new DenseTensor<long>(MemoryMarshal.Cast<byte,long>(bytes).ToArray(),shape);
                feeds.Add(pair.Name,t); inputBits.Add(pair.Name,Support.Hash(bytes));
            }
            var context = graph.CreateExecution(ExecutionOptions.Memory);
            Support.Require(context.Execute(feeds,true,ExecutionProvider.CPU,ExecutionOptions.Memory),context.LastErrorMessage ?? "Encoder failed");
            foreach (var pair in c.GetProperty("outputs").EnumerateObject())
                Support.Require(Support.Bytes(context.Outputs[pair.Name]!).AsSpan().SequenceEqual(File.ReadAllBytes(Support.PathOf(root,pair.Value))),"Complete encoder output changed: "+name);
            foreach (var pair in feeds) Support.Require(Support.Hash(Support.Bytes(pair.Value)) == inputBits[pair.Key],"Input mutated");
            foreach (var node in nodes)
            {
                var a = (Tensor<float>)context.GetInputTensor(node.Inputs[0]); var b = (Tensor<float>)graph.Initializers[node.Inputs[1]];
                var y = (Tensor<float>)context.GetInputTensor(node.Outputs[0]);
                Support.Require(a.Dimensions.SequenceEqual(new[] { 1,frames,b.Dimensions[0] }) && y.Dimensions.SequenceEqual(new[] { 1,frames,b.Dimensions[1] }),"Actual operand geometry");
                if (!weights.ContainsKey(node.Inputs[1])) weights.Add(node.Inputs[1],Save("weight-"+node.ID+".bin",b));
                entries.Add(new { name, node = node.ID, node_name = node.Name, m = frames, k = b.Dimensions[0], n = b.Dimensions[1], weight = node.Inputs[1],
                    a = Save(index+"-a.bin",a), y = Save(index+"-y.bin",y) }); index++;
            }
            requests.Add(new { name, frames, complete_encoder_matches = true, inputs_unchanged = true });
            context.Reset(); Console.WriteLine(name+" complete encoder and operands verified");
        }
        foreach (var pair in initializers) Support.Require(Support.Hash(Support.Bytes(graph.Initializers[pair.Key])) == pair.Value,"Weight mutated");
        Support.Require(index == 12,"Coverage");
        Support.Write(Path.Combine(output,"result.json"),new { passed = true, identity = Support.Identity(), requests, weights, entries, taps,
            scope = "Four complete encoder calls; declared diagnostic outputs retain selected operands under Memory policy; original full outputs agree bitwise" });
    }
}
