using System;
using System.Buffers;
using System.Numerics;
using System.Runtime.InteropServices;

namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    // Adapt the output-lane panel idea of voice 03af0bf. Preserve master's
    // separate multiply and add, increasing reduction order and gate epilogue.
    // Packing is per invocation, so mutable weights cannot leave stale panels.
    sealed class LstmProjectionPanels : IDisposable
    {
        readonly float[] storage;
        readonly int inputSize, hiddenSize, outputs, inputElements;

        LstmProjectionPanels(float[] storage, int inputSize, int hiddenSize, int directions)
        {
            this.storage = storage;
            this.inputSize = inputSize;
            this.hiddenSize = hiddenSize;
            outputs = checked(4 * hiddenSize);
            inputElements = checked(directions * outputs * inputSize);
        }

        internal static LstmProjectionPanels? Create(ReadOnlySpan<float> w, ReadOnlySpan<float> r,
            int inputSize, int hiddenSize, int directions, int sequence, TensorExecutionOptions options)
        {
            if (!options.UseSimd || !Vector.IsHardwareAccelerated || sequence < 8 || hiddenSize < 16 || hiddenSize > 128)
                return null;
            int count = StorageLength(w.Length, r.Length);
            if (count == 0) return null;
            float[] storage = ArrayPool<float>.Shared.Rent(count);
            try
            {
                options.ScratchReporter?.AddScratchBytes((long)count * sizeof(float));
                int outputs = checked(4 * hiddenSize);
                Transpose(w, storage.AsSpan(0, w.Length), inputSize, outputs, directions);
                Transpose(r, storage.AsSpan(w.Length, r.Length), hiddenSize, outputs, directions);
                return new LstmProjectionPanels(storage, inputSize, hiddenSize, directions);
            }
            catch
            {
                ArrayPool<float>.Shared.Return(storage);
                throw;
            }
        }

        static void Transpose(ReadOnlySpan<float> source, Span<float> target, int inputs, int outputs, int directions)
        {
            int length = checked(inputs * outputs);
            for (int d = 0; d < directions; d++)
            for (int k = 0; k < inputs; k++)
            for (int o = 0; o < outputs; o++)
                target[d * length + k * outputs + o] = source[d * length + o * inputs + k];
        }

        internal void Input(int direction, ReadOnlySpan<float> x, Span<float> y) =>
            LstmProjectOrdered(x, storage.AsSpan(direction * inputSize * outputs, inputSize * outputs), y);

        internal void InputBlock(int direction, ReadOnlySpan<float> input, int start, int stride, int rows, Span<float> output) =>
            LstmProjectOrderedRows(input, start, stride, inputSize,
                storage.AsSpan(direction * inputSize * outputs, inputSize * outputs), output, rows);

        internal void Recurrent(int direction, ReadOnlySpan<float> x, Span<float> y) =>
            LstmProjectOrdered(x, storage.AsSpan(inputElements + direction * hiddenSize * outputs, hiddenSize * outputs), y);

        public void Dispose() => ArrayPool<float>.Shared.Return(storage);

        internal static int StorageLength(int inputWeights, int recurrentWeights)
        {
            if (inputWeights < 0 || recurrentWeights < 0) return 0;
            long count = (long)inputWeights + recurrentWeights;
            return count <= Array.MaxLength ? (int)count : 0;
        }
    }

    internal static void LstmProjectOrdered(ReadOnlySpan<float> input, ReadOnlySpan<float> panel, Span<float> output)
    {
        if (panel.Length != checked(input.Length * output.Length))
            throw new ArgumentException("The projection panel does not match its input and output dimensions.");
        int width = Vector<float>.Count, block = 4 * width, o = 0;
        ref float weights = ref MemoryMarshal.GetReference(panel);
        ref float destination = ref MemoryMarshal.GetReference(output);
        if (Vector.IsHardwareAccelerated)
        {
            for (; o <= output.Length - block; o += block)
            {
                var a = Vector<float>.Zero; var b = a; var c = a; var d = a;
                for (int k = 0; k < input.Length; k++)
                {
                    var x = new Vector<float>(input[k]);
                    nuint row = (nuint)(k * output.Length + o);
                    a = Vector.Add(a, Vector.Multiply(x, Vector.LoadUnsafe(ref weights, row)));
                    b = Vector.Add(b, Vector.Multiply(x, Vector.LoadUnsafe(ref weights, row + (nuint)width)));
                    c = Vector.Add(c, Vector.Multiply(x, Vector.LoadUnsafe(ref weights, row + (nuint)(2 * width))));
                    d = Vector.Add(d, Vector.Multiply(x, Vector.LoadUnsafe(ref weights, row + (nuint)(3 * width))));
                }
                a.StoreUnsafe(ref destination, (nuint)o);
                b.StoreUnsafe(ref destination, (nuint)(o + width));
                c.StoreUnsafe(ref destination, (nuint)(o + 2 * width));
                d.StoreUnsafe(ref destination, (nuint)(o + 3 * width));
            }
        }
        for (; o < output.Length; o++)
        {
            float value = 0f;
            for (int k = 0; k < input.Length; k++) value += input[k] * panel[k * output.Length + o];
            output[o] = value;
        }
    }
}
