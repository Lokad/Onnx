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

    public T? Attr<T, U>(string name, T? d = default(T))
    {
        if (Attributes is null) return d;
        if (HasAttr<T>(name)) return Attr<T>(name);
        if (!HasAttr<U>(name))
        {
            return d;
        }
        else
        {
            var u = Attr<U>(name);
            return (T?) Convert.ChangeType(u, typeof(T)) ?? throw new Exception("Cannot convert attribute to type " + typeof(T).ToString());
        }

    }

    public T RequiredAttr<T, U>(string name)
    {
        if (Attributes is null) throw new ArgumentException("The attribute " + name + " is required for node execution but was not found.");
        if (HasAttr<T>(name))
        {
            return RequiredAttr<T>(name);
        }
        else if (HasAttr<U>(name))
        {
            var u = Attr<U>(name);
            return (T)Convert.ChangeType(u, typeof(T))!;
        }
        else
        {
            throw new ArgumentException("The attribute " + name + " is required for node execution but was not found.");
        }
    }

    public int? Int(string name) => Attr<int?, long?>(name);

    public int RequiredInt(string name) => RequiredAttr<int, long>(name);

    public T[]? ArrayAttr<T, U>(string name)   
    {
        if (Attributes is null) return null;
        var t = Attr<T[]>(name);
        if (t is not null) return t;
        var u = Attr<U[]>(name);  
        if (u is null)
        {
            return null;  
        }
        else
        {
            return u.Select(e => (T) (Convert.ChangeType(e, typeof(T)) ?? throw new Exception("Cannot convert array element to type " + typeof(T).ToString()))).ToArray(); 
        }

    }

    public int[]? Ints(string name) => ArrayAttr<int, long>(name);

    public ITensor? InputTensor(ComputationalGraph graph, int index) => index < Inputs.Length ? graph.GetInputTensor(Inputs[index]) : null;

    public OpResult Execute(ComputationalGraph graph, ExecutionProvider provider = ExecutionProvider.CPU) 
    {
        try
        {
            if (provider == ExecutionProvider.CPU) 
            {
                var r =  ExecuteCPU(graph);
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

        OpType.Conv => CPU.Conv(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), 
            Attr<string>("auto_pad"), Attr<int[]>("dilations"), Attr<int?>("group"), Attr<int[]>("kernel_shape"), Attr<int[]>("pads"), Attr<int[]>("strides")),
        
        OpType.Relu => CPU.Relu(InputTensor(graph, 0)),

        OpType.Erf => CPU.Erf(InputTensor(graph, 0)),

        OpType.MaxPool => CPU.MaxPool(InputTensor(graph, 0), Attr<string>("auto_pad"), Attr<int?>("ceil_mode"), ArrayAttr<int, long>("dilations"), ArrayAttr<int, long>("kernel_shape"), ArrayAttr<int, long>("pads"), Attr<int?>("storage_order"), ArrayAttr<int, long>("strides")),

        OpType.MatMul => CPU.MatMul(InputTensor(graph, 0), InputTensor(graph, 1)),

        OpType.Squeeze => CPU.Squeeze(graph.GetOpVersion(), graph.GetInputTensor(Inputs[0]), Attr<ITensor>("axes")),

        OpType.Transpose => CPU.Transpose(graph.GetInputTensor(Inputs[0]), Ints("perm")),

        OpType.Constant => CPU.Constant(OneOfAttr("sparse_value", "value", "value_float", "value_floats", "value_int", "value_ints", "value_string", "value_strings")),

        OpType.Cast => CPU.Cast(graph.GetInputTensor(Inputs[0]), RequiredInt("to")),

        OpType.Concat => CPU.Concat(graph.GetInputTensors(Inputs), RequiredInt("axis")),
        
        _ => NotSupported(Op)
    };
}

