"""One layout-only change over the qualified source; leave root source untouched."""
PREFIX = 'src/Lokad.Onnx/'
SOURCE_FILES = [PREFIX + name for name in ['GraphLstmPacking.cs',
    'CPUExecutionProvider.LstmPanels.cs', 'CPUExecutionProvider.Recurrent.cs']]
HELPER = PREFIX + 'PreparedLstmProjection.cs'
TARGETS = [SOURCE_FILES[0], SOURCE_FILES[2], HELPER]


def changed(files):
    result = dict(files)
    packing, panels, recurrent = SOURCE_FILES
    original = result[packing]
    old = '''        for (int k = 0; k < HiddenSize; k++)
        for (int o = 0; o < 4 * HiddenSize; o++)
            values[k * 4 * HiddenSize + o] = input[o * HiddenSize + k];'''
    new = '''        int block = PreparedLstmProjection.ColumnsPerBlock;
        for (int o = 0; o < 4 * HiddenSize; o += block)
        {
            int columns = Math.Min(block, 4 * HiddenSize - o);
            for (int k = 0; k < HiddenSize; k++)
            for (int lane = 0; lane < columns; lane++)
                values[o * HiddenSize + k * columns + lane] = input[(o + lane) * HiddenSize + k];
        }'''
    assert original.count(old) == 1
    original = original.replace(old, new)
    original = original.replace('An independently owned [direction, input, gate] transpose of a constant LSTM weight.',
        'Independently owned constant LSTM weights grouped by four output vectors.')
    result[packing] = original
    original = result[panels]
    start = original.index('    internal static void LstmProjectOrdered(')
    assert original.endswith('    }\n}\n')
    helper = original[start:-2]
    helper = helper.replace('void LstmProjectOrdered(', 'void Multiply(')
    helper = helper.replace('block = 4 * width', 'block = ColumnsPerBlock')
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
    result[HELPER] = """namespace Lokad.Onnx;

using System;
using System.Numerics;
using System.Runtime.InteropServices;

internal static class PreparedLstmProjection
{
    // Prepared arrays stay within one process; Vector.Count is fixed for its lifetime.
    // Keep the original reader's four accumulators and arithmetic.
    internal static int ColumnsPerBlock => 4 * Vector<float>.Count;

""" + helper + '}\n' 
    original = result[recurrent]
    for before in ['LstmProjectOrdered(xs.Slice(xOff, inputSize), preparedInput, xw)',
                   'LstmProjectOrdered(hv, preparedRecurrent, hr)']:
        assert original.count(before) == 1
        original = original.replace(before, before.replace('LstmProjectOrdered', 'PreparedLstmProjection.Multiply'))
    result[recurrent] = original
    assert {name for name in result if result[name] != files.get(name)} == set(TARGETS)
    return result
