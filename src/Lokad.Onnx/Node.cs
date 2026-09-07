namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

using static Lokad.Onnx.OpResult;
using CPU = CPUExecutionProvider;

public partial struct Node
{
    public long ID;
    public string Name;
    public Dictionary<string, object>? Attributes;
    public OpType Op;
    public string[] Inputs;
    public string[] Outputs;

    public bool HasAttr<T>(string name) => Attributes is not null && Attributes.ContainsKey(name) && Attributes[name].GetType() == typeof(T);

    public T? Attr<T>(string name, T? d = default(T)) => Attributes is not null && Attributes.ContainsKey(name) && Attributes[name].GetType() == typeof(T) ?
        (T)Attributes[name] : d;

    public T RequiredAttr<T>(string name) => Attributes is not null && Attributes.ContainsKey(name) && Attributes[name].GetType() == typeof(T) ?
    (T)Attributes[name] : throw new ArgumentException("The attribute " + name + " is required for node execution but was not found.");

    public object? OneOfAttr(params string[] names)
    {
        var a = Attributes;
        if (a is null) return null;
        var name = names.FirstOrDefault(n => a.ContainsKey(n));
        return name is null ? null : a[name];
    }

    public int? Int(string name, int? d = null)
    {
        if (Attributes is null) return d;
        if (HasAttr<int>(name)) return Attr<int>(name);
        if (!HasAttr<long>(name))
        {
            return d;
        }
        else
        {
            var u = Attr<long>(name);
            return (int?)Convert.ToInt32(u) ?? throw new Exception("Cannot convert attribute to type int.");
        }
    }

    public int RequiredInt(string name) => Int(name) ?? throw new ArgumentException($"The Int attribute {name} is required but was not found.");

    public int[]? Ints(string name)
    {
        if (Attributes is null) return null;
        if (HasAttr<int[]>(name))
        {
            return Attr<int[]>(name);
        }
        else if (HasAttr<long[]>(name))
        {
            var u = Attr<long[]>(name);
            return u!.Select(e => Convert.ToInt32(e)).ToArray();
        }
        else
        {
            return null;
        }

    }

    public int[] RequiredInts(string name) => Ints(name) ?? throw new ArgumentException($"The Ints attribute {name} is required but was not found.");


    public ITensor? InputTensor(ComputationalGraph graph, int index) =>
        index < Inputs.Length && !string.IsNullOrEmpty(Inputs[index]) ? graph.GetInputTensor(Inputs[index]) : null;

    public ITensor? InputTensorOrAttr(ComputationalGraph graph, int index, string name) => index < Inputs.Length ? graph.GetInputTensor(Inputs[index]) : Attr<ITensor>(name);

    public OpResult Execute(ComputationalGraph graph, ExecutionProvider provider = ExecutionProvider.CPU, ExecutionOptions? options = null)
    {
        try
        {
            if (provider == ExecutionProvider.CPU)
            {
 
                var r = ExecuteCPU(graph, options);
                
                if (r.Status == OpStatus.Success && r.Outputs.Length != Outputs.Length)
                {
                    return Failure(Op, $"The operation returned {r.Outputs.Length} outputs but the graph node has {Outputs.Length} outputs.");
                }
                else
                {
                    for (int i = 0; i < r.Outputs.Length; i++)
                    {
                        r.Outputs[i].Name = this.Outputs[i];
                    }
                    return r;
                }
            }
            else
            {
                throw new NotSupportedException();
            }
        }
        catch (ArgumentNullException ane)
        {
            return !string.IsNullOrEmpty(ane.ParamName) ? MissingInput(Op, ane.ParamName) : Failure(Op, ane.Message);
        }
        catch (TensorInputShapeException tise)
        {
            return WrongInputShape(Op, tise.Name, tise.Shape, tise.Input);
        }
        catch (Exception e)
        {
            return Failure(Op, e.Message);
        }
    }

    public OpResult ExecuteCPU(ComputationalGraph graph, ExecutionOptions? options = null)
    {
        var opt = options ?? graph.Options;
        return Op switch
    {
        OpType.Reshape => CPU.Reshape(InputTensor(graph, 0), InputTensor(graph, 1), Attr<bool?>("allow_zero"), opt),

        OpType.Add => CPU.Add(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Sub => CPU.Sub(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Mul => CPU.Mul(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Div => CPU.Div(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Pow => CPU.Pow(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Sqrt => CPU.Sqrt(InputTensor(graph, 0), opt),

        OpType.Conv => CPU.Conv(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2),
            Attr<string>("auto_pad"), Ints("dilations"), Attr<int?>("group"), Ints("kernel_shape"), Ints("pads"), Ints("strides")),

        OpType.Relu => CPU.Relu(InputTensor(graph, 0), opt),

        OpType.Erf => CPU.Erf(InputTensor(graph, 0), opt, graph.ActivePool),

        OpType.MaxPool => CPU.MaxPool(InputTensor(graph, 0), Attr<string>("auto_pad"), Attr<int?>("ceil_mode"), Ints("dilations"), Ints("kernel_shape"), Ints("pads"), Attr<int?>("storage_order"), Ints("strides"), opt),

        OpType.MatMul => CPU.MatMul(InputTensor(graph, 0), InputTensor(graph, 1), opt, graph.ActivePool),

        OpType.Transpose => CPU.Transpose(InputTensor(graph, 0), Ints("perm"), opt, graph.ActivePool),

        OpType.Constant => CPU.Constant(OneOfAttr("sparse_value", "value", "value_float", "value_floats", "value_int", "value_ints", "value_string", "value_strings"), opt),

        OpType.Cast => CPU.Cast(InputTensor(graph, 0), RequiredInt("to"), opt),

        OpType.Concat => CPU.Concat(graph.GetInputTensors(Inputs), RequiredInt("axis"), opt),

        OpType.Shape => CPU.Shape(InputTensor(graph, 0), Int("start"), Int("end"), opt),

        OpType.Gather => CPU.Gather(InputTensor(graph, 0), InputTensor(graph, 1), Int("axis"), opt),

        OpType.Slice => CPU.Slice(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), InputTensor(graph, 3), InputTensor(graph, 4), opt),

        OpType.Equal => CPU.Equal(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Where => CPU.Where(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), opt),

        OpType.Expand => CPU.Expand(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Resize => CPU.Resize(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), InputTensor(graph, 3),
            Attr<string>("mode", "nearest"), Attr<string>("coordinate_transformation_mode", "half_pixel"), Attr<string>("nearest_mode", "round_prefer_floor"),
            Attr<float>("cubic_coeff_a", -0.75f), Attr<float>("extrapolation_value", 0f)),

        OpType.Unsqueeze => graph.OpsetVersion() switch
        {
            int v when v >= 13 => CPU.Unsqueeze(InputTensor(graph, 0), InputTensor(graph, 1), opt),
            _ => CPU.Unsqueeze(InputTensor(graph, 0), RequiredInts("axes"), opt),
        }, 

        OpType.ReduceSum => CPU.ReduceSum(InputTensor(graph, 0), InputTensor(graph, 1), Int("keepdims"), Int("noop_with_empty_axes"), opt),

        OpType.ReduceMean => graph.OpsetVersion() switch
        {
            int v when v >= 18 => CPU.ReduceMean(InputTensor(graph, 0), InputTensor(graph, 1), Int("keepdims"), Int("noop_with_empty_axes"), opt),
            _ => CPU.ReduceMean(InputTensor(graph, 0), Ints("axes")?.ToTensor<int>(), Int("keepdims"), Int("noop_with_empty_axes"), opt),
        },
        
        OpType.ReduceMax => graph.OpsetVersion() switch { int v when v >= 18 => CPU.ReduceMax(InputTensor(graph, 0), InputTensor(graph, 1), Int("keepdims"), opt), _ => CPU.ReduceMax(InputTensor(graph, 0), Ints("axes")?.ToTensor<int>(), Int("keepdims"), opt), },

        OpType.Softmax => CPU.Softmax(InputTensor(graph, 0), Int("axis"), opt, graph.ActivePool),

        OpType.Abs => CPU.Abs(InputTensor(graph, 0), opt),

        OpType.Cos => CPU.Cos(InputTensor(graph, 0), opt),

        OpType.Sin => CPU.Sin(InputTensor(graph, 0), opt),

        OpType.Neg => CPU.Neg(InputTensor(graph, 0), opt),

        OpType.Gelu => CPU.Gelu(InputTensor(graph, 0), Attr<string>("approximate"), opt, graph.ActivePool),

        OpType.Squeeze => CPU.Squeeze(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.Range => CPU.Range(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), opt),

        OpType.Tile => CPU.Tile(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.LayerNormalization => CPU.LayerNormalization(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), Int("axis"), Attr<float>("epsilon"), opt, graph.ActivePool),

        OpType.SplitToSequence => CPU.SplitToSequence(InputTensor(graph, 0), InputTensor(graph, 1), Int("axis"), Int("keepdims"), opt),

        OpType.SequenceAt => CPU.SequenceAt(InputTensor(graph, 0), InputTensor(graph, 1), opt),

        OpType.RotaryEmbedding => CPU.RotaryEmbedding(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), RequiredInt("half"), Int("axis"), Int("concatAxis"), opt, graph.ActivePool),

        _ => NotSupported(Op)
    };
    }
}

