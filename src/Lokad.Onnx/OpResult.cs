using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Lokad.Onnx
{
    public enum OpStatus
    {
        Success,
        Failure
    }

    public struct OpResult
    {
        #region Fields
        public OpType Op;
        public OpStatus Status;
        public string? Message = null;
        /// <summary>Original exception behind a failure, when one was captured; null otherwise.</summary>
        public Exception? Cause = null;
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

        public static OpResult Failure(OpType op, string message) => Failure(op, message, null);
        public static OpResult Failure(OpType op, string message, Exception? cause) =>
          new OpResult(op, OpStatus.Failure) { Message = message, Cause = cause };
        public static OpResult NotSupported(OpType op) =>
            new OpResult(op, OpStatus.Failure) { Message = $"The operation {op} is not supported by the backend." };
        public static OpResult InputTypeNotSupported(OpType op, string name, ITensor input) => InputTypeNotSupported(op, name, input, null);
        public static OpResult InputTypeNotSupported(OpType op, string name, ITensor input, string? msg) =>
           new OpResult(op, OpStatus.Failure) { Message = $"The type {input.ElementType} for input parameter {name} is not supported for operation {op}. " + msg ?? ""};
        public static OpResult AttributeNotSupported(OpType op, string name, string attr, string? msg) =>
            new OpResult(op, OpStatus.Failure) { Message = $"The attribute {name} with value {attr} is not supported for operation {op}. " + msg ?? ""};

        public static OpResult MissingInput(OpType op, string name) => Failure(op, $"The required input parameter {name} is missing or null.");

        public static OpResult MissingAttribute(OpType op, string name, string? msg) => Failure(op, $"The required attribute {name} is missing or null. " + msg ?? "");

        public static OpResult WrongInputType(OpType op, string name, TensorElementType type, ITensor input) => WrongInputType(op, name, type, input, null);
        public static OpResult WrongInputType(OpType op, string name, TensorElementType type, ITensor input, string? msg) => Failure(op, $"The input tensor {input.Name} for parameter {name} has type {input.ElementType} but is required to be {type.ToString()}. " + msg ?? "");

        public static OpResult WrongInputType(OpType op, string name, string message, ITensor input) => Failure(op, $"The input tensor {input.Name} for parameter {name} has the wrong type: {message}.");

        public static OpResult WrongInputShape(OpType op, string name, int rank, ITensor input) => Failure(op, $"The input tensor {input.Name} for parameter {name} has shape {input.PrintShape()} but is required to have rank {rank}.");

        public static OpResult WrongInputShape(OpType op, string name, int[] dims, ITensor input) => Failure(op, $"The input tensor {input.Name} for parameter {name} has shape {input.PrintShape()} but is required to be {dims.Print()}.");

        public static OpResult WrongInputShape(OpType op, string name, ITensor input, string msg) => Failure(op, $"The input tensor {input.Name} for parameter {name} has shape {input.PrintShape()}. {msg}.");

        public static OpResult CannotBroadcast(OpType op, ITensor x, ITensor y) => Failure(op, $"The tensors {x.TensorNameDesc()} and {y.TensorNameDesc()} are not compatible for broadcasting.");

        #endregion
    }
}
