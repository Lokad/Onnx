"""Extract exact paired softmax and change only final maximum/validity reductions."""
from pathlib import Path
import argparse, hashlib, importlib.util, json


def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True); a = p.parse_args()
    root = Path(__file__).resolve().parents[3]
    source = root / 'tests/e5/softmax-zero-blocks/generate.py'
    spec = importlib.util.spec_from_file_location('source_extractor', source)
    parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
    tensor = root / 'src/Lokad.Onnx/TensorOps.Elementwise.cs'; math = root / 'src/Lokad.Onnx/MathOps.cs'
    ts = tensor.read_text(encoding='utf-8'); ms = math.read_text(encoding='utf-8')
    original_max = parser.method(ts, 'SoftmaxContiguousMaxMasked').replace('SoftmaxContiguousMaxMasked', 'OriginalMax')
    candidate_max = original_max.replace('OriginalMax', 'ReducedMax')
    loop = '            for (int l = 1; l < width; l++) if (vmax[l] > max) max = vmax[l];'
    unrolled = '            if (width == 8)\n            {\n'
    for lane in range(1, 8):
        unrolled += f'                if (vmax[{lane}] > max) max = vmax[{lane}];\n'
    unrolled += '            }\n            else\n    ' + loop
    assert candidate_max.count(loop) == 1
    candidate_max = candidate_max.replace(loop, unrolled)
    flags = '            for (int l = 0; l < width; l++) if (vok[l] == 0) return float.NaN;'
    assert candidate_max.count(flags) == 1
    candidate_max = candidate_max.replace(flags, '            if (!Vector.EqualsAll(vok, new Vector<int>(-1))) return float.NaN;')
    exp = parser.method(ms, 'ExpVectorNonpositive').replace('ExpVectorNonpositive', 'OriginalExp')
    body = parser.method(ts, 'SoftmaxMaskedFloatSpanPtr')
    start = body.index('            bool useWideExp ='); end = body.index(';', start) + 1
    body = body[:start] + body[end:]; removed = 0
    while 'if (useWideExp)' in body:
        start = body.index('if (useWideExp)'); end = parser.closing(body, body.index('{', start))
        body = body[:start] + body[end:]; removed += 1
    assert removed == 3 and body.count('MathOps.ExpVectorSoftmax(') == 3
    body = body.replace('MathOps.ExpVectorSoftmax(', 'OriginalExp(')
    copied = body.replace('SoftmaxMaskedFloatSpanPtr', 'Original').replace('SoftmaxContiguousMaxMasked', 'OriginalMax')
    candidate = body.replace('SoftmaxMaskedFloatSpanPtr', 'Reduced').replace('SoftmaxContiguousMaxMasked', 'ReducedMax')
    code = 'using System;\nusing System.Numerics;\nusing System.Runtime.CompilerServices;\nnamespace SoftmaxReduction;\ninternal static class Kernels\n{\n'
    code += '[MethodImpl(MethodImplOptions.AggressiveInlining)]\n' + exp + '\n' + original_max + '\n' + candidate_max + '\n' + copied + '\n' + candidate + '\n}\n'
    a.output.mkdir(parents=True, exist_ok=False)
    path = a.output / 'Kernels.cs'; path.write_text(code, encoding='utf-8')
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    pins = dict(source={str(p.relative_to(root)): sha(p) for p in (tensor, math, source)},
                generator_sha256=sha(Path(__file__)), kernel_sha256=sha(path),
                changed_maximum_loop=1, changed_validity_loop=1, removed_unselected_wide_branches=3,
                scope='Standalone nonpositive-on/wide-off default kernel; no zero-block or pointer-max combination')
    (a.output / 'source.json').write_text(json.dumps(pins, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__': main()
