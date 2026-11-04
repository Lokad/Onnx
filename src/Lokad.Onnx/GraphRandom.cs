namespace Lokad.Onnx;

using System;
using System.Runtime.CompilerServices;
using System.Threading;

public partial class ComputationalGraph
{
    // Output-name arrays identify nodes across value-type Node copies. Allocate only
    // for random operators. Explicit contexts own streams; transient facade contexts
    // borrow their serialized owner's stream, so Execute/Reset do not reseed it.
    ConditionalWeakTable<string[], NormalRandomState>? normalRandomStates;
    ComputationalGraph? randomStateOwner;

    internal void BorrowRandomState(ComputationalGraph owner) => randomStateOwner = owner;

    internal NormalRandomState NormalRandomState(string[] outputs) => randomStateOwner is not null
        ? randomStateOwner.NormalRandomState(outputs)
        : LazyInitializer.EnsureInitialized(ref normalRandomStates).GetValue(outputs, static _ => new NormalRandomState());
}

internal sealed class NormalRandomState
{
    Random? random;
    int? seed;
    bool hasSpare;
    double spare;

    // The provider holds this object's monitor for a complete draw.
    internal void SetSeed(int? value)
    {
        if (random is not null && seed == value) return;
        random = new Random(value ?? Random.Shared.Next());
        seed = value;
        hasSpare = false;
    }

    internal double Next()
    {
        if (hasSpare)
        {
            hasSpare = false;
            return spare;
        }
        double radius = Math.Sqrt(-2 * Math.Log(1 - random!.NextDouble()));
        var (sin, cos) = Math.SinCos(2 * Math.PI * random.NextDouble());
        spare = radius * sin;
        hasSpare = true;
        return radius * cos;
    }
}
