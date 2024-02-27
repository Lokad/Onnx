namespace Lokad.Onnx;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;

using static Lokad.Onnx.OpResult;
using CPU = CPUExecutionProvider;

public struct Node
{
    public long ID;
    public string Name;
    public Dictionary<string, object>? Attributes;
    public Satsuma.Node WeightedGraphNode;
    public OpType Op;
    public string[] Inputs;
    public string[] Outputs;

    public T? Attr<T>(string name) => Attributes is not null && Attributes.ContainsKey(name) && Attributes[name].GetType() == typeof(T) ? 
        (T)Attributes[name] : default(T);

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
        
        OpType.Conv => CPU.Conv(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), 
            Attr<string>("auto_pad"), Attr<int[]>("dilations"), Attr<int?>("group"), Attr<int[]>("kernel_shape"), Attr<int[]>("pads"), Attr<int[]>("strides")),
        
        OpType.Relu => CPU.Relu(InputTensor(graph, 0)),

        OpType.MaxPool => CPU.MaxPool(InputTensor(graph, 0), Attr<string>("auto_pad"), Attr<int?>("ceil_mode"), ArrayAttr<int, long>("dilations"), ArrayAttr<int, long>("kernel_shape"), ArrayAttr<int, long>("pads"), Attr<int?>("storage_order"), ArrayAttr<int, long>("strides")),

        OpType.MatMul => CPU.MatMul(InputTensor(graph, 0), InputTensor(graph, 1)),

        OpType.Squeeze => CPU.Squeeze(graph.GetOpVersion(), graph.GetInputTensor(Inputs[0]), Attr<ITensor>("axes")),
        
        _ => NotSupported(Op)
    };
}

