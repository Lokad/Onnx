"""One layout-only change over the qualified source; leave root source untouched."""
PREFIX = 'src/Lokad.Onnx/'
TARGETS = [PREFIX + name for name in ['GraphLstmPacking.cs',
    'CPUExecutionProvider.LstmPanels.cs', 'CPUExecutionProvider.Recurrent.cs']]


def changed(files):
    result = dict(files)
    packing, panels, recurrent = TARGETS
    original = result[packing]
    old = '''        for (int k = 0; k < HiddenSize; k++)
        for (int o = 0; o < 4 * HiddenSize; o++)
            values[k * 4 * HiddenSize + o] = input[o * HiddenSize + k];'''
    new = '''        int block = ColumnsPerBlock;
        for (int o = 0; o < 4 * HiddenSize; o += block)
        {
            int columns = Math.Min(block, 4 * HiddenSize - o);
            for (int k = 0; k < HiddenSize; k++)
            for (int lane = 0; lane < columns; lane++)
                values[o * HiddenSize + k * columns + lane] = input[(o + lane) * HiddenSize + k];
        }'''
    anchor = '    internal const int HiddenSize = 640;'
    addition = '''    // Prepared arrays stay within one process; Vector.Count is fixed for its lifetime.
    // The consumer keeps the same four accumulators as LstmProjectOrdered.
    internal static int ColumnsPerBlock => 4 * System.Numerics.Vector<float>.Count;

'''
    assert original.count(old) == original.count(anchor) == 1
    original = original.replace(old, new).replace(anchor, addition + anchor)
    original = original.replace('An independently owned [direction, input, gate] transpose of a constant LSTM weight.',
        'Independently owned constant LSTM weights grouped by four output vectors.')
    result[packing] = original
    original = result[panels]
    start = original.index('    internal static void LstmProjectOrdered(')
    assert original.endswith('    }\n}\n')
    helper = original[start:-2]
    helper = helper.replace('void LstmProjectOrdered(', 'void LstmProjectPreparedOrdered(')
    helper = helper.replace('block = 4 * width', 'block = GraphLstmPacking.ColumnsPerBlock')
    assert helper.count('k * output.Length + o') == 2
    helper = helper.replace('nuint row = (nuint)(k * output.Length + o);',
        'nuint row = (nuint)(o * input.Length + k * block);')
    helper = helper.replace('''            float value = 0f;
            for (int k = 0; k < input.Length; k++) value += input[k] * panel[k * output.Length + o];''',
        '''            int group = o / block * block;
            int columns = Math.Min(block, output.Length - group);
            float value = 0f;
            for (int k = 0; k < input.Length; k++)
                value += input[k] * panel[group * input.Length + k * columns + o - group];''')
    assert 'k * output.Length + o' not in helper
    result[panels] = original[:-2] + '\n' + helper + '}\n'
    original = result[recurrent]
    for before in ['LstmProjectOrdered(xs.Slice(xOff, inputSize), preparedInput, xw)',
                   'LstmProjectOrdered(hv, preparedRecurrent, hr)']:
        assert original.count(before) == 1
        original = original.replace(before, before.replace('LstmProjectOrdered', 'LstmProjectPreparedOrdered'))
    result[recurrent] = original
    assert {name for name in result if result[name] != files[name]} == set(TARGETS)
    return result
