namespace Lokad.Onnx;

using System;
using System.Numerics;

/// <summary>
/// A broadcast elementwise operator with scalar and vector forms. Operator
/// structs (one per arithmetic operation) let the broadcast kernels call
/// both forms directly, without per-element or per-vector delegate calls.
/// </summary>
public interface IBroadcastOperator<T> where T : unmanaged
{
    static abstract T Scalar(T left, T right);
    static abstract Vector<T> Vector(Vector<T> left, Vector<T> right);
}

/// <summary>Broadcast addition.</summary>
public readonly struct AddBroadcast<T> : IBroadcastOperator<T> where T : unmanaged, INumber<T>
{
    public static T Scalar(T left, T right) => left + right;
    public static Vector<T> Vector(Vector<T> left, Vector<T> right) => left + right;
}

/// <summary>Broadcast subtraction.</summary>
public readonly struct SubtractBroadcast<T> : IBroadcastOperator<T> where T : unmanaged, INumber<T>
{
    public static T Scalar(T left, T right) => left - right;
    public static Vector<T> Vector(Vector<T> left, Vector<T> right) => left - right;
}

/// <summary>Broadcast multiplication.</summary>
public readonly struct MultiplyBroadcast<T> : IBroadcastOperator<T> where T : unmanaged, INumber<T>
{
    public static T Scalar(T left, T right) => left * right;
    public static Vector<T> Vector(Vector<T> left, Vector<T> right) => left * right;
}

/// <summary>Broadcast division.</summary>
public readonly struct DivideBroadcast<T> : IBroadcastOperator<T> where T : unmanaged, INumber<T>
{
    public static T Scalar(T left, T right) => left / right;
    public static Vector<T> Vector(Vector<T> left, Vector<T> right) => left / right;
}
