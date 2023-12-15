using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public enum OpType
    {
        Squeeze,
        MatAdd
    }
    
    public enum OpStatus
    {
        Success,
        Failure
    }

    public struct OpResult
    {
        public OpType Op;
        public OpStatus Status;
        public string? Message = null;
        ITensor[] Inputs = { };
        ITensor[] Outputs = { };

        public OpResult(OpType op, OpStatus status)
        {  
           Op = op; 
           Status = status; 
        }  

        public static OpResult NotSupported(OpType op) => 
            new OpResult(op, OpStatus.Failure) { Message = $"The operation {op} is not supported." };

        public static OpResult NotSupported(OpType op, string pname, TensorElementType type) =>
            new OpResult(op, OpStatus.Failure) { Message = $"The operation {op} is not supported for input paramer {pname} type {type}." };
        public static OpResult WrongInputParameterType(OpType op, TensorElementType ptype, ITensor input) =>
            new OpResult(op, OpStatus.Failure) { Message = $"The input parameter {input.Name} has type {ptype} not {input.ElementType}." };
        public static OpResult Failure(OpType op, string message) =>
           new OpResult(op, OpStatus.Failure) { Message = message };
    }

    public struct Node
    {
        public long ID;
        public string Name;
        public string Description;
        public Satsuma.Node WeightedGraphNode;
        public OpType Op;
        public string[] Inputs;
        public string[] Outputs;
    }
}
