"""Apply only bounded input-row batching to the selected recurrent provider."""
import difflib
import hashlib
from pathlib import Path

PANELS = 'src/Lokad.Onnx/CPUExecutionProvider.LstmPanels.cs'
RECURRENT = 'src/Lokad.Onnx/CPUExecutionProvider.Recurrent.cs'
HELPER = 'src/Lokad.Onnx/Zzz.LstmInputBlocks.cs'


def once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def transform(panels, recurrent):
    panels, recurrent = panels.replace('\r\n','\n'), recurrent.replace('\r\n','\n')
    assert hashlib.sha256(panels.encode()).hexdigest() == '308e904a89549641e61a6c443db9ac63e88c624407a7a860b4c23d50e69fcf5f'
    assert hashlib.sha256(recurrent.encode()).hexdigest() == '061c845e32cd5f7f543036c81084e3459a3ea9445f09a24f905bf4c3fab4e826'
    p = once(panels, '        internal void Recurrent(int direction,', '''        internal void InputBlock(int direction, ReadOnlySpan<float> input, int start, int stride, int rows, Span<float> output) =>
            LstmProjectOrderedRows(input, start, stride, inputSize,
                storage.AsSpan(direction * inputSize * outputs, inputSize * outputs), output, rows);

        internal void Recurrent(int direction,''')
    r = once(recurrent, '        var xw = new float[4 * hiddenSize];', '        Span<float> xw = new float[4 * hiddenSize];')
    needle = '        using var projections = LstmProjectionPanels.Create(ws, rs, inputSize, H, numDirections, seq, opts.Tensor);'
    r = once(r, needle, needle+'''
        // At most four rows of four gates: <= 8 KiB under existing panel admission.
        var inputBlock = projections is null ? null : new float[16 * H];
        if (inputBlock is not null) opts.Tensor.ScratchReporter?.AddScratchBytes((long)inputBlock.Length * sizeof(float));''')
    r = once(r, '                        projections.Input(d, xs.Slice(xOff, inputSize), xw);', '''                        if (s % 4 == 0)
                        {
                            int rows = Math.Min(4, limit - s);
                            int stride = checked(batch * inputSize);
                            projections.InputBlock(d, xs, xOff, rev ? -stride : stride, rows, inputBlock.AsSpan(0, rows * 4 * H));
                        }
                        xw = inputBlock.AsSpan((s % 4) * 4 * H, 4 * H);''')
    helper = (Path(__file__).parent/'OrderedRows.cs.txt').read_text(encoding='utf-8-sig')
    texts = {PANELS:p, RECURRENT:r, HELPER:helper}
    patch = ''
    for name,old,new in [(PANELS,panels,p),(RECURRENT,recurrent,r),(HELPER,'',helper)]:
        patch += ''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='a/'+name if old else '/dev/null',tofile='b/'+name))
    # The recurrent projection and complete gate loop remain textually unchanged.
    gate = '                    for (int h = 0; h < H; h++)'
    assert r[r.index(gate):] == recurrent[recurrent.index(gate):]
    assert 'projections.Recurrent(d, hv, hr);' in r
    return texts, patch
