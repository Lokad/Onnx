"""Reversible instrumentation of only the seven unmapped calls per round."""
EDITS = [
    ('        public long SetupTicks;', '        public long SetupTicks;\n        public long[]? CallStarts, CallTicks;'),
    ('                SetupTicks = Stopwatch.GetTimestamp() - start });',
     '                CallStarts = kind == "unmapped" ? new long[780 * batch] : null,\n'
     '                CallTicks = kind == "unmapped" ? new long[780 * batch] : null,\n'
     '                SetupTicks = Stopwatch.GetTimestamp() - start });'),
    ('            for (int call = 0; call < f.Batch; call++) f.Returned[call] = Tensor<float>.MatMul(f.A, f.B, f.Options);',
     '            if (f.Kind == "unmapped")\n'
     '            {\n'
     '                int offset = iteration * f.Batch;\n'
     '                for (int call = 0; call < f.Batch; call++)\n'
     '                {\n'
     '                    long callStart = Stopwatch.GetTimestamp();\n'
     '                    f.Returned[call] = Tensor<float>.MatMul(f.A, f.B, f.Options);\n'
     '                    f.CallTicks![offset + call] = Stopwatch.GetTimestamp() - callStart;\n'
     '                    f.CallStarts![offset + call] = callStart;\n'
     '                }\n'
     '            }\n'
     '            else for (int call = 0; call < f.Batch; call++) f.Returned[call] = Tensor<float>.MatMul(f.A, f.B, f.Options);'),
    ('                exact = true, inputs = true, ownership = true, clocks = f.Clocks });',
     '                exact = true, inputs = true, ownership = true, clocks = f.Clocks, call_starts = f.CallStarts, call_ticks = f.CallTicks });'),
    ('            calls = 780 * batches, warmups = 600 * batches, measured = 180 * batches, rows },',
     '            calls = 780 * batches, warmups = 600 * batches, measured = 180 * batches, rows, diagnostic_only = true, release_admitted = false },')]


def changed(original):
    value = original
    for before, after in EDITS:
        assert value.count(before) == 1, before
        value = value.replace(before, after)
    restored = value
    for before, after in reversed(EDITS):
        assert restored.count(after) == 1
        restored = restored.replace(after, before)
    assert restored == original
    return value
