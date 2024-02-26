namespace Lokad.Onnx.Backend;

using System;
using System.Collections.Generic;
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
                return Execute(graph, provider);
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
        OpType.Resize => CPU.Reshape(InputTensor(graph, 0), InputTensor(graph, 1), Attr<bool>("allowZero")),
        OpType.Squeeze => CPU.Squeeze(graph.GetOpVersion(), graph.GetInputTensor(Inputs[0]), Attr<ITensor>("axes")),
        _ => throw new NotSupportedException(),
    };
}

