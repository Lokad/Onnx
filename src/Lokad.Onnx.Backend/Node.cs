namespace Lokad.Onnx.Backend;

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

using static Lokad.Onnx.Backend.OpResult;
using CPU = CPUExecutionProvider;

[RequiresPreviewFeatures]
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
        OpType.Reshape => CPU.Reshape(InputTensor(graph, 0), InputTensor(graph, 1), Attr<bool>("allow_zero")),
        
        OpType.Add => CPU.Add(InputTensor(graph, 0), InputTensor(graph, 1)),
        
        OpType.Conv => CPU.Conv(InputTensor(graph, 0), InputTensor(graph, 1), InputTensor(graph, 2), 
            Attr<string>("auto_pad"), Attr<int[]>("dilations"), Attr<int?>("group"), Attr<int[]>("kernel_shape"), Attr<int[]>("pads"), Attr<int[]>("strides")),
        
        OpType.Relu => CPU.Relu(InputTensor(graph, 0)),
        
        OpType.Squeeze => CPU.Squeeze(graph.GetOpVersion(), graph.GetInputTensor(Inputs[0]), Attr<ITensor>("axes")),
        _ => NotSupported(Op)
    };
}

