
using System;
using System.Collections.Generic;
using System.Runtime.Versioning;

using CPU = Lokad.Onnx.Backend.CPUExecutionProvider;

namespace Lokad.Onnx.Backend
{
    public enum OpStatus
    {
        Success,
        Failure
    }

    [RequiresPreviewFeatures]
    public struct OpResult
    {
        #region Fields
        public OpType Op;
        public OpStatus Status;
        public string? Message = null;
        public ITensor[] Inputs = {};
        public ITensor[]? Outputs = null;
        #endregion

        #region Constructors
        public OpResult(OpType op, OpStatus status)
        {  
           Op = op; 
           Status = status; 
        }
        #endregion

        #region Methods
        public static OpResult NotSupported(OpType op) => 
            new OpResult(op, OpStatus.Failure) { Message = $"The operation {op} is not supported." };

        public static OpResult NotSupported(OpType op, string pname, TensorElementType type) =>
            new OpResult(op, OpStatus.Failure) { Message = $"The operation {op} is not supported for input paramer {pname} type {type}." };
        
        public static OpResult WrongInputParameterType(OpType op, TensorElementType ptype, ITensor input) =>
            new OpResult(op, OpStatus.Failure) { Message = $"The input parameter {input.Name} has type {ptype} not {input.ElementType}." };
        
        public static OpResult Failure(OpType op, string message) =>
           new OpResult(op, OpStatus.Failure) { Message = message };

        public static OpResult MissingInput(OpType op, string name) => Failure(op, $"The input parameter {name} is missing or null.");

        public static OpResult WrongInputType(OpType op, string name, TensorElementType type, ITensor input) => Failure(op, $"The input tensor {input.Name} for parameter {name} has type {input.ElementType} but is required to be .");

        public static OpResult Success(OpType op, params ITensor[] output) =>
           new OpResult(op, OpStatus.Success) { Outputs=output };
        #endregion
    }

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

        public OpResult MissingInput(string name) => OpResult.Failure(Op, $"The input parameter {name} is missing or null.");

        public OpResult WrongInputType(string name, TensorElementType type, ITensor input) => OpResult.Failure(Op, $"The input tensor {input.Name} for parameter {name} has type {input.ElementType} but is required to be .");


        public OpResult Execute(ComputationalGraph graph, ExecutionProvider provider = ExecutionProvider.CPU) => provider switch
        {
            ExecutionProvider.CPU => ExecuteCPU(graph),
            _ => throw new NotSupportedException(),
        };

        public OpResult ExecuteCPU(ComputationalGraph graph) => Op switch
        {
            OpType.Resize => CPU.Reshape(InputTensor(graph, 0), InputTensor(graph, 1), Attr<bool>("allowZero")),
            OpType.Squeeze => CPU.Squeeze(graph.GetOpVersion(), graph.GetInputTensor(Inputs[0]), Attr<ITensor>("axes")),
            _ => throw new NotSupportedException(),
        };
    }
}
