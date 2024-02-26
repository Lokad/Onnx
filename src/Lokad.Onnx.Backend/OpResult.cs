using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Versioning;
using System.Text;
using System.Threading.Tasks;

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
        public ITensor[] Inputs = { };
        public ITensor[] Outputs = { };
        #endregion

        #region Constructors
        public OpResult(OpType op, OpStatus status)
        {
            Op = op;
            Status = status;
        }
        #endregion

        #region Methods
        public static OpResult Success(OpType op, params ITensor[] outputs) =>
           new OpResult(op, OpStatus.Success) { Outputs = outputs };

        public static OpResult Failure(OpType op, string message) =>
          new OpResult(op, OpStatus.Failure) { Message = message };
        public static OpResult NotSupported(OpType op) =>
            new OpResult(op, OpStatus.Failure) { Message = $"The operation {op} is not supported." };

        public static OpResult NotSupported(OpType op, string pname, TensorElementType type) =>
            new OpResult(op, OpStatus.Failure) { Message = $"The operation {op} is not supported for input paramer {pname} type {type}." };

        public static OpResult WrongInputParameterType(OpType op, TensorElementType ptype, ITensor input) =>
            new OpResult(op, OpStatus.Failure) { Message = $"The input parameter {input.Name} has type {ptype} not {input.ElementType}." };



        public static OpResult MissingInput(OpType op, string name) => Failure(op, $"The input parameter {name} is missing or null.");

        public static OpResult WrongInputType(OpType op, string name, TensorElementType type, ITensor input) => Failure(op, $"The input tensor {input.Name} for parameter {name} has type {input.ElementType} but is required to be .");

        public static OpResult WrongInputShape(OpType op, string name, int[] dims, ITensor input) => Failure(op, $"The input tensor {input.Name} for parameter {name} has shape {input.PrintShape()} but is required to be {dims.Print()}.");


        #endregion
    }
}
