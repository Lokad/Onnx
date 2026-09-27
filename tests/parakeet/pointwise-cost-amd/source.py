"""Reversible observations in three methods of the actual qualified root."""
CONV = 'src/Lokad.Onnx/TensorOps.ConvPool.cs'
MATMUL = 'src/Lokad.Onnx/TensorOps.MatMul.cs'
HELPER = 'src/Lokad.Onnx/PointwiseCostProbe.cs'


def edits(name):
    if name == CONV:
        return [
            ('        var dimensions = new int[] { N, M, outH, outW };',
             '        PointwiseCostProbe.Enter(input, weight, N, C, H, W, M, group, kH, kW, dH, dW, sH, sW, pad.top, pad.left, pad.bottom, pad.right, outH, outW, bias is not null, options);\n'
             '        PointwiseCostProbe.Start(PointwiseCostProbe.Stage.OutputInitialize);\n'
             '        var dimensions = new int[] { N, M, outH, outW };'),
            ('        if (output.Length == 0) return output;\n        var xd = input.ToDenseTensor();',
             '        PointwiseCostProbe.End();\n        if (output.Length == 0) return output;\n'
             '        PointwiseCostProbe.Start(PointwiseCostProbe.Stage.Materialize);\n        var xd = input.ToDenseTensor();'),
            ('        PointwiseCostProbe.Start(PointwiseCostProbe.Stage.Materialize);\n        var xd = input.ToDenseTensor();\n        var wd = weight.ToDenseTensor();\n        var bd = bias?.ToDenseTensor();\n        int inBatch = C * H * W;',
             '        PointwiseCostProbe.Start(PointwiseCostProbe.Stage.Materialize);\n        var xd = input.ToDenseTensor();\n        var wd = weight.ToDenseTensor();\n        var bd = bias?.ToDenseTensor();\n        PointwiseCostProbe.End();\n'
             '        PointwiseCostProbe.Materialized(input, weight, xd, wd);\n        int inBatch = C * H * W;'),
            ('            RunPointwiseBatchesFloat(xMem, wMem, bMem, hasBias, oMem, N, group, C, H, W, M, outH, outW, inBatch, outBatch, options);\n            return output;',
             '            RunPointwiseBatchesFloat(xMem, wMem, bMem, hasBias, oMem, N, group, C, H, W, M, outH, outW, inBatch, outBatch, options);\n'
             '            PointwiseCostProbe.Exit();\n            return output;')]
    assert name == MATMUL
    old = '''                        PackPanelsB(n, k, y, pp);
                        if (!AblationSwitches.EnablePackedAvx512Dynamic || n < 64
                            || !TryPackedAvx512Rows(blocked, n, k, x, pp, output))
                            mm_unsafe_vectorized_intrinsics_2x4packed_bump(blocked, n, k, x, pp, output);'''
    new = '''                        PointwiseCostProbe.Scratch(packed.Length);
                        PointwiseCostProbe.Start(PointwiseCostProbe.Stage.Pack);
                        PackPanelsB(n, k, y, pp);
                        PointwiseCostProbe.End();
                        if (!AblationSwitches.EnablePackedAvx512Dynamic || n < 64
                            || !TryPackedAvx512Rows(blocked, n, k, x, pp, output))
                        {
                            PointwiseCostProbe.Kernel(blocked, n, k);
                            mm_unsafe_vectorized_intrinsics_2x4packed_bump(blocked, n, k, x, pp, output);
                            PointwiseCostProbe.End();
                        }'''
    return [(old, new),
        ('        if (clearDestination) destination.Buffer.Span.Clear();',
         '        PointwiseCostProbe.Start(PointwiseCostProbe.Stage.Clear);\n'
         '        if (clearDestination) destination.Buffer.Span.Clear();\n        PointwiseCostProbe.End();')]


def changed(name, raw):
    original = raw.decode().replace('\r\n', '\n')
    value = original
    tail = ''
    if name == MATMUL:
        marker = '    public static Tensor<double> MatMul2D(Tensor<double> x, Tensor<double> y) =>'
        assert value.count(marker) == 1
        value, remainder = value.split(marker)
        tail = marker + remainder
    for before, after in edits(name):
        assert value.count(before) == 1, (name, before)
        value = value.replace(before, after)
    restored = value
    for before, after in reversed(edits(name)):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored + tail == original
    value += tail
    return (value.replace('\n', '\r\n') if b'\r\n' in raw else value).encode()
