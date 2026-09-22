"""Add wide independent-output loops; retain selected portable implementations."""
import difflib
import re

PANELS='src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs'
ROWS='src/Lokad.Onnx/Zzz.LstmInputBlocks.cs'
WIDE='src/Lokad.Onnx/Zzz.LstmWideProjection.cs'


def body(source,start):
    opening=source.index('{',start);depth=1;end=opening+1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    return source[start:end]


def replace_once(source,old,new):
    assert source.count(old)==1,old
    return source.replace(old,new)


def widen(method,rows):
    old='LstmProjectOrderedRows' if rows else 'LstmProjectOrdered'
    method=replace_once(method,old+'(',old+'512(')
    if rows:method=method.replace('LstmProjectOrdered(', 'LstmProjectOrdered512(')
    loop=body(method,method.index('for (; o <= '+('vectorEnd' if rows else 'output.Length')+' - block;'))
    expanded=loop.replace('Vector<float>','Vector512<float>').replace('Vector.','Vector512.')
    expanded=expanded.replace('new Vector512<float>(', 'Vector512.Create(')
    expanded=re.sub(r'\bwidth\b','wideWidth',expanded)
    expanded=re.sub(r'\bblock\b','wideBlock',expanded)
    insertion='''if (Vector512.IsHardwareAccelerated)
        {
            const int wideWidth = 16, wideBlock = '''+('32' if rows else '64')+''';
            '''+expanded+'''
        }
        '''
    return replace_once(method,'if (Vector.IsHardwareAccelerated)',insertion+'if (Vector.IsHardwareAccelerated)')


def transform(panels,rows):
    original=panels
    single=body(panels,panels.index('    internal static void LstmProjectOrdered('))
    multiple=body(rows,rows.index('    internal static void LstmProjectOrderedRows('))
    wide='''using System;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Lokad.Onnx;

public partial class CPUExecutionProvider
{
    // Independent outputs widen; each lane keeps increasing-k separate multiply/add.
'''+widen(single,False)+'\n\n'+widen(multiple,True)+'\n}\n'
    panels=replace_once(panels,'        readonly float[] storage;','        readonly float[] storage;\n        bool useWide;')
    panels=replace_once(panels,'return new LstmProjectionPanels(storage, inputSize, hiddenSize, directions);',
        '''return new LstmProjectionPanels(storage, inputSize, hiddenSize, directions)
                {
                    useWide = options.UseIntrinsics && System.Runtime.Intrinsics.Vector512.IsHardwareAccelerated
                        && System.Runtime.Intrinsics.X86.Avx512F.IsSupported
                };''')
    replacements=[
        ('''        internal void Input(int direction, ReadOnlySpan<float> x, Span<float> y) =>
            LstmProjectOrdered(x, storage.AsSpan(direction * inputSize * outputs, inputSize * outputs), y);''',
         '''        internal void Input(int direction, ReadOnlySpan<float> x, Span<float> y)
        {
            var panel = storage.AsSpan(direction * inputSize * outputs, inputSize * outputs);
            if (useWide) LstmProjectOrdered512(x, panel, y);
            else LstmProjectOrdered(x, panel, y);
        }'''),
        ('''        internal void InputBlock(int direction, ReadOnlySpan<float> input, int start, int stride, int rows, Span<float> output) =>
            LstmProjectOrderedRows(input, start, stride, inputSize,
                storage.AsSpan(direction * inputSize * outputs, inputSize * outputs), output, rows);''',
         '''        internal void InputBlock(int direction, ReadOnlySpan<float> input, int start, int stride, int rows, Span<float> output)
        {
            var panel = storage.AsSpan(direction * inputSize * outputs, inputSize * outputs);
            if (useWide) LstmProjectOrderedRows512(input, start, stride, inputSize, panel, output, rows);
            else LstmProjectOrderedRows(input, start, stride, inputSize, panel, output, rows);
        }'''),
        ('''        internal void Recurrent(int direction, ReadOnlySpan<float> x, Span<float> y) =>
            LstmProjectOrdered(x, storage.AsSpan(inputElements + direction * hiddenSize * outputs, hiddenSize * outputs), y);''',
         '''        internal void Recurrent(int direction, ReadOnlySpan<float> x, Span<float> y)
        {
            var panel = storage.AsSpan(inputElements + direction * hiddenSize * outputs, hiddenSize * outputs);
            if (useWide) LstmProjectOrdered512(x, panel, y);
            else LstmProjectOrdered(x, panel, y);
        }''')]
    for old,new in replacements:panels=replace_once(panels,old,new)
    assert body(panels,panels.index('    internal static void LstmProjectOrdered('))==single
    assert 'FusedMultiplyAdd' not in wide
    diff=''.join(difflib.unified_diff(original.splitlines(True),panels.splitlines(True),fromfile=PANELS,tofile=PANELS))
    diff+=''.join(difflib.unified_diff([],wide.splitlines(True),fromfile='/dev/null',tofile=WIDE))
    return {PANELS:panels,WIDE:wide},diff
