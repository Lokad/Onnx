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
    public Satsuma.Node WeightedGraphNode;
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

    public ITensor? InputTensor(ComputationalGraph graph, int index) => index < Inputs.Length ? graph.GetInputTensor(Inputs[index]) : null;

    public OpResult Execute(ComputationalGraph graph, ExecutionProvider provider = ExecutionProvider.CPU)
    {
        try
        {
            if (provider == ExecutionProvider.CPU)
            {
                var r = ExecuteCPU(graph);
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

    public OpResult ExecuteCPU(ComputationalGraph graph) => Op switch
    {
        OpType.Reshape => CPU.Reshape(InputTensor(graph, 0), InputTensor(graph, 1), Attr<bool?>("allow_zero")),

        OpType.Add => CPU.Add(InputTensor(graph, 0), InputTensor(graph, 1)),

        OpType.Sub => CPU.Sub(InputTensor(graph, 0), InputTensor(graph, 1)),

        OpType.Mul => CPU.Mul(InputTensor(graph, 0), InputTensor(graph, 1)),

        OpType.Div => CPU.Div(InputTensor(graph, 0), InputTensor(graph, 1)),

        OpType.Pow => CPU.Pow(InputTensor(graph, 0), InputTensor(graph, 1)),

        OpType.Conv => CPU.Conv(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2),
            Attr<string>("auto_pad"), Attr<int[]>("dilations"), Attr<int?>("group"), Attr<int[]>("kernel_shape"), Attr<int[]>("pads"), Attr<int[]>("strides")),

        OpType.Relu => CPU.Relu(InputTensor(graph, 0)),

        OpType.Erf => CPU.Erf(InputTensor(graph, 0)),

        OpType.MaxPool => CPU.MaxPool(InputTensor(graph, 0), Attr<string>("auto_pad"), Attr<int?>("ceil_mode"), Ints("dilations"), Ints("kernel_shape"), Ints("pads"), Attr<int?>("storage_order"), Ints("strides")),

        OpType.MatMul => CPU.MatMul(InputTensor(graph, 0), InputTensor(graph, 1)),

        OpType.Squeeze => CPU.Squeeze(graph.GetOpVersion(), graph.GetInputTensor(Inputs[0]), Attr<ITensor>("axes")),

        OpType.Transpose => CPU.Transpose(graph.GetInputTensor(Inputs[0]), Ints("perm")),

        OpType.Constant => CPU.Constant(OneOfAttr("sparse_value", "value", "value_float", "value_floats", "value_int", "value_ints", "value_string", "value_strings")),

        OpType.Cast => CPU.Cast(graph.GetInputTensor(Inputs[0]), RequiredInt("to")),

        OpType.Concat => CPU.Concat(graph.GetInputTensors(Inputs), RequiredInt("axis")),

        OpType.Shape => CPU.Shape(graph.GetInputTensor(Inputs[0]), Int("start"), Int("end")),

        OpType.Gather => CPU.Gather(graph.GetInputTensor(Inputs[0]), graph.GetInputTensor(Inputs[1]), Int("axis")),

        OpType.Slice => CPU.Slice(graph.GetInputTensor(Inputs[0]), graph.GetInputTensor(Inputs[1]), graph.GetInputTensor(Inputs[2]), graph.GetInputTensor(Inputs, 3), graph.GetInputTensor(Inputs, 4)),

        OpType.Unsqueeze => CPU.Unsqueeze(graph.GetInputTensor(Inputs[0]), graph.GetInputTensor(Inputs[1])),

        OpType.ReduceSum => CPU.ReduceSum(graph.GetInputTensor(Inputs[0]), graph.GetInputTensor(Inputs[1]), Int("keepdims"), Int("noop_with_empty_axes")),

        OpType.ReduceMean => CPU.ReduceMean(graph.GetInputTensor(Inputs[0]), graph.GetInputTensor(Inputs[1]), Int("keepdims"), Int("noop_with_empty_axes")),

        OpType.ReduceMax => CPU.ReduceMax(graph.GetInputTensor(Inputs[0]), graph.GetInputTensor(Inputs[1]), Int("keepdims")),

        OpType.Softmax => CPU.Softmax(graph.GetInputTensor(Inputs[0]), Int("axis")),

        _ => NotSupported(Op)
    };
}

