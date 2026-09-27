"""Reversible hooks in two qualified methods; arithmetic leaves stay exact."""
PROVIDER = 'src/Lokad.Onnx/CPUExecutionProvider.MatMul.cs'
MATMUL = 'src/Lokad.Onnx/Zzz.IsolatedShortMatMul.cs'
HELPER = 'src/Lokad.Onnx/AttentionCostProbe.cs'


def edits(name):
    if name == PROVIDER:
        return [('        var opts = (options ?? ExecutionOptions.Default).Validated();\n        switch (A.ElementType)',
                 '        var opts = (options ?? ExecutionOptions.Default).Validated();\n'
                 '        using var observation = AttentionCostProbe.Enter(A, B, opts.Tensor);\n'
                 '        switch (A.ElementType)')]
    assert name == MATMUL
    return [
        ('        float[] packed = RentScratch<float>(n * k, options);',
         '        AttentionCostProbe.Route(m, n, k, rows);\n'
         '        AttentionCostProbe.Start(AttentionCostProbe.Stage.Rent);\n'
         '        float[] packed = RentScratch<float>(n * k, options);\n'
         '        AttentionCostProbe.End();\n'
         '        AttentionCostProbe.Scratch(packed.Length);'),
        ('                ShortWidePackPanelsB(n, k, y, pp);',
         '                AttentionCostProbe.Start(AttentionCostProbe.Stage.Pack);\n'
         '                ShortWidePackPanelsB(n, k, y, pp);\n'
         '                AttentionCostProbe.End();\n'
         '                AttentionCostProbe.Start(AttentionCostProbe.Stage.Arithmetic);'),
        ('                    if (threeRows)\n                        ShortWideMultiply3Rows(rows, n, k, x, pp, output);\n'
         '                    else\n                        ShortWideMultiply2Rows(rows, n, k, x, pp, output);',
         '                    if (threeRows)\n                    {\n'
         '                        AttentionCostProbe.Leaf("ShortWideMultiply3Rows");\n'
         '                        ShortWideMultiply3Rows(rows, n, k, x, pp, output);\n                    }\n'
         '                    else\n                    {\n'
         '                        AttentionCostProbe.Leaf("ShortWideMultiply2Rows");\n'
         '                        ShortWideMultiply2Rows(rows, n, k, x, pp, output);\n                    }'),
        ('                }\n            }\n        }\n        finally',
         '                }\n                AttentionCostProbe.End();\n            }\n        }\n        finally'),
        ('            ArrayPool<float>.Shared.Return(packed);',
         '            AttentionCostProbe.Start(AttentionCostProbe.Stage.Return);\n'
         '            ArrayPool<float>.Shared.Return(packed);\n            AttentionCostProbe.End();'),
        ('        if (rows != m)\n            ShortWideMultiplyRemainder(1, n, k, x + rows * n, y, output + rows * k);',
         '        if (rows != m)\n        {\n'
         '            AttentionCostProbe.Start(AttentionCostProbe.Stage.FinalRow);\n'
         '            ShortWideMultiplyRemainder(1, n, k, x + rows * n, y, output + rows * k);\n'
         '            AttentionCostProbe.End();\n        }')]


def changed(name, raw):
    original = raw.decode().replace('\r\n', '\n')
    value = original
    for before, after in edits(name):
        assert value.count(before) == 1, (name, before)
        value = value.replace(before, after)
    restored = value
    for before, after in reversed(edits(name)):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == original
    return (value.replace('\n', '\r\n') if b'\r\n' in raw else value).encode()
