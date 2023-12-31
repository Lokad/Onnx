namespace Lokad.Onnx;

using System.Numerics;
/// <summary>
/// Represents a unary operator that operates on a single value or vector.
/// </summary>
/// <typeparam name="T">The type of the value or vector.</typeparam>
public interface IUnaryOperator<T>
{
    /// <summary>
    /// Applies the unary operator to the specified value.
    /// </summary>
    /// <param name="x">The value to apply the operator to.</param>
    /// <returns>The result of applying the operator to the value.</returns>
    T Invoke(T x);

    /// <summary>
    /// Applies the unary operator to the specified vector.
    /// </summary>
    /// <param name="x">The vector to apply the operator to.</param>
    /// <returns>The result of applying the operator to the vector.</returns>
    //Vector<T> Invoke(Vector<T> x);
}

/// <summary>
/// Represents a binary operator that operates on two values or vectors.
/// </summary>
/// <typeparam name="T">The type of the values or vectors.</typeparam>
public interface IBinaryOperator<T>
    where T : struct
{
    /// <summary>
    /// Applies the binary operator to the specified values.
    /// </summary>
    /// <param name="x">The first value to apply the operator to.</param>
    /// <param name="y">The second value to apply the operator to.</param>
    /// <returns>The result of applying the operator to the values.</returns>
    T Invoke(T x, T y);

    /// <summary>
    /// Applies the binary operator to the specified vectors.
    /// </summary>
    /// <param name="x">The first vector to apply the operator to.</param>
    /// <param name="y">The second vector to apply the operator to.</param>
    /// <returns>The result of applying the operator to the vectors.</returns>
    Vector<T> Invoke(Vector<T> x, Vector<T> y);
}

/// <summary>
/// Represents a ternary operator that operates on three values or vectors.
/// </summary>
/// <typeparam name="T">The type of the values or vectors.</typeparam>
public interface ITernaryOperator<T>
    where T : struct
{
    /// <summary>
    /// Applies the ternary operator to the specified values.
    /// </summary>
    /// <param name="x">The first value to apply the operator to.</param>
    /// <param name="y">The second value to apply the operator to.</param>
    /// <param name="z">The third value to apply the operator to.</param>
    /// <returns>The result of applying the operator to the values.</returns>
    T Invoke(T x, T y, T z);

    /// <summary>
    /// Applies the ternary operator to the specified vectors.
    /// </summary>
    /// <param name="x">The first vector to apply the operator to.</param>
    /// <param name="y">The second vector to apply the operator to.</param>
    /// <param name="z">The third vector to apply the operator to.</param>
    /// <returns>The result of applying the operator to the vectors.</returns>
    Vector<T> Invoke(Vector<T> x, Vector<T> y, Vector<T> z);
}

