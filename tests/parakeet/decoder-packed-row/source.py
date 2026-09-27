"""One fixed dispatch patch over the qualified release; no root mutation."""
TARGET = 'src/Lokad.Onnx/TensorOps.MatMul.cs'
HELPER = 'src/Lokad.Onnx/PreparedSingleRowKernel.cs'


def changed(original):
    before = '''        if ((m & 1) != 0 && (m % 3) != 0) return null;'''
    after = '''        if (m == 1)
        {
            // Consume an existing preparation only in the current wide one-row territory.
            if (y.Dimensions[^1] < M1BlockedMinColumns) return null;
        }
        else if ((m & 1) != 0 && (m % 3) != 0) return null;'''
    entry = '''    static unsafe void RunPreparedPackedRows(int m, int n, int k, float* x, float* packed, float* dest)
    {'''
    routed = entry + '''
        if (m == 1)
        {
            PreparedSingleRowKernel.Multiply(n, k, x, packed, dest);
            return;
        }'''
    assert original.count(before) == original.count(entry) == 1
    result = original.replace(before, after).replace(entry, routed)
    assert result.replace(after, before).replace(routed, entry) == original
    return result
