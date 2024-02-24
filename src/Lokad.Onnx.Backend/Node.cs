
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;

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

        public static OpResult Success(OpType op, ITensor[]? output = null, string? message = null) =>
           new OpResult(op, OpStatus.Success) { Outputs=output, Message = message };
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

        public bool HasAttr<T>(string name) => Attributes is not null && Attributes.ContainsKey(name) ? true : false;

        public T Attr<T>(string name) => Attributes is not null && Attributes.ContainsKey(name) && Attributes[name].GetType() == typeof(T) ? 
            (T)Attributes[name] : throw new ArgumentException(name, $"The node {Name} does not have the attribute {name} of type {typeof(T)}.");
        
        public OpResult Execute(ComputationalGraph graph, ExecutionProvider provider = ExecutionProvider.CPU) => provider switch
        {
            ExecutionProvider.CPU => ExecuteCPU(graph),
            _ => throw new NotSupportedException(),
        };

        public OpResult ExecuteCPU(ComputationalGraph graph) => Op switch
        {
            OpType.Squeeze => CPUExecutionProvider.Squeeze(graph.GetOpVersion(), graph.GetTensor(Inputs[0]), Attr<ITensor>("axes")),
            _ => throw new NotSupportedException(),
        };
    }
}
