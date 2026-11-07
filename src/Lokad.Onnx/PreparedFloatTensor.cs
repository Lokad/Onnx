namespace Lokad.Onnx;

using System;
using System.Threading;

// Private prepared values keep independent mutable storage for each use.
// No prepared array may escape through the public API.
internal sealed class PreparedFloatValue
{
    readonly float[] values;
    readonly int[] shape;

    internal PreparedFloatValue(DenseTensor<float> value)
    {
        values = value.Buffer.ToArray();
        shape = value.Dimensions.ToArray();
    }

    internal PreparedFloatTensor CreateTensor(ICopyAccountant? reporter = null) => new(new PreparedFloatTensor.Owner(values, reporter), shape);
    internal bool IsUntouched(ITensor value) => value is PreparedFloatTensor tensor
        && tensor.IsPrepared(values) && tensor.Dimensions.SequenceEqual(shape);
}

internal sealed class PreparedFloatTensor : Tensor<float>
{
    internal sealed class Owner
    {
        readonly float[] prepared;
        readonly ICopyAccountant? reporter;
        float[]? writable;
        internal Owner(float[] prepared, ICopyAccountant? reporter = null) { this.prepared = prepared; this.reporter = reporter; }
        internal float[] Read => Volatile.Read(ref writable) ?? prepared;
        internal bool HasCopy => Volatile.Read(ref writable) is not null;
        internal bool IsPrepared(float[] source) => ReferenceEquals(prepared, source) && !HasCopy;
        internal float[] Write()
        {
            var current = Volatile.Read(ref writable);
            if (current is not null) return current;
            var copy = (float[])prepared.Clone();
            reporter?.AddCopyBytes(4L * copy.Length);
            return Interlocked.CompareExchange(ref writable, copy, null) ?? copy;
        }
    }

    readonly Owner owner;
    internal PreparedFloatTensor(Owner owner, ReadOnlySpan<int> shape) : base(shape, false)
    {
        if (Length != owner.Read.Length) throw new ArgumentException("Reshape must preserve the element count.", nameof(shape));
        this.owner = owner;
    }

    internal bool HasMaterialized => owner.HasCopy;
    internal bool IsPrepared(float[] source) => owner.IsPrepared(source);
    internal Memory<float> WritableMemory => owner.Write();

    // This view is restricted to audited numerical consumers which never
    // mutate an input or return it as an output. It is never publicly bound.
    internal DenseTensor<float> BorrowReadOnly() => new(owner.Read, Dimensions) { Name = Name };

    public override float GetValue(int index) => owner.Read[index];
    public override void SetValue(int index, float value) => owner.Write()[index] = value;
    public override Tensor<float> Reshape(ReadOnlySpan<int> shape) => new PreparedFloatTensor(owner, shape);
    public override DenseTensor<float> ToDenseTensor() => new(owner.Write(), Dimensions) { Name = Name };
    public override Tensor<float> Clone() => new DenseTensor<float>((float[])owner.Read.Clone(), Dimensions);
    public override Tensor<TResult> CloneEmpty<TResult>(ReadOnlySpan<int> shape) => new DenseTensor<TResult>(shape);
    protected override void CopyTo(float[] array, int arrayIndex)
    {
        ValidateCopyTo(array, arrayIndex, nameof(arrayIndex));
        owner.Read.CopyTo(array, arrayIndex);
    }
}
