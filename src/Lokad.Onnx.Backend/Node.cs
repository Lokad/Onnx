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
        OpStatus Status;
        string? Message = null;
        ITensor[] Inputs = { };
        ITensor[] Outputs = { };

        public OpResult(OpType op, OpStatus status)
        {  
           Op = op; 
           Status = status; 
        }  

        public static OpResult NotSupported(OpType op) => 
            new OpResult(op, OpStatus.Failure) { Message = $"The operation {op} is not supported." };
        
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
