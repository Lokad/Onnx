namespace Lokad.Onnx;

using System;
using System.Collections.Generic;

/// <summary>
/// Name-to-value bindings for graph inputs and outputs. A slot holds null
/// while it is declared but unbound, exactly like uncomputed entries in the
/// intermediate map. Reading an unbound or missing slot through the indexer
/// throws naming it; TryGetValue exposes the marker for explicit unresolved
/// checks. String comparison stays ordinal, matching the previous maps.
/// </summary>
public sealed class BindingMap : Dictionary<string, ITensor?>
{
    public BindingMap() : base(StringComparer.Ordinal)
    {
    }

    public BindingMap(IDictionary<string, ITensor?> source) : base(source, StringComparer.Ordinal)
    {
    }

    /// <summary>Bound value for the slot.</summary>
    /// <exception cref="KeyNotFoundException">The slot name is not declared.</exception>
    /// <exception cref="InvalidOperationException">The slot is declared but has no value in this run.</exception>
    public new ITensor this[string key]
    {
        get => base[key] ?? throw new InvalidOperationException($"Graph binding {key} is declared but has no value in this run.");
        set => base[key] = value;
    }

    /// <summary>Marks the slot declared but unbound without a value.</summary>
    internal void MarkUnresolved(string name) => base[name] = null;
}
