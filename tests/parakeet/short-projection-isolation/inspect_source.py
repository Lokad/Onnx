"""Inspect the rejected global flags and a bounded isolation mechanism; no build."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
OLD = ROOT / 'artifacts/parakeet-first-use-kernels-source-20260923'
APP = ROOT / 'artifacts/parakeet-first-use-kernels-app-amd-20260923'
GRAPHS = ROOT / 'artifacts/warmed-release-amd-v2-20260923'
METHODS = ['PackPanelsB', 'mm_unsafe_vectorized_intrinsics_2x4packed_bump',
           'mm_unsafe_vectorized_intrinsics_3x4packed', 'mm_unsafe_vectorized_intrinsics']


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def method(text, header):
    start = text.index(header); opened = text.index('{', start); depth = 1; end = opened + 1
    while depth:
        depth += (text[end] == '{') - (text[end] == '}'); end += 1
    return text[start:end], text[:start].count('\n') + 1


def main():
    assert not (OUT / 'source-observations-20260923.json').exists()
    assert pin(OLD / 'prepared.json')['sha256'] == '829e26d7acd55ccf969f4292949abc19385a014f562517344a3265d42a0f51c0'
    prepared = read(OLD / 'prepared.json')
    assert len(prepared['before']) == 420 and len(prepared['source']) == 421
    for name, wanted in prepared['before'].items(): assert pin(ROOT / name) == wanted, name
    for name, wanted in prepared['source'].items(): assert pin(OLD / 'source' / name) == wanted, name
    for folder, digest in [(APP, 'adb3a270880bb4f30ac87fec92c2f6e18495d56c8ed8cffcfb58a950f7a25128'),
                           (GRAPHS, '921e1303a193f0f57b2ccd47fd98f6c378d26c617a10d271e6ab4f94b901b175')]:
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        assert pin(folder / 'analysis.json') == proof['files']['analysis.json']
    assert read(APP / 'analysis.json')['performance']['admitted']
    assert read(APP / 'closed.json')['admitted'] and not read(GRAPHS / 'closed.json')['admitted']
    name = 'src/Lokad.Onnx/MathOps.cs'
    source = (ROOT / name).read_text(); old = (OLD / 'source' / name).read_text()
    attribute = '    [MethodImpl(MethodImplOptions.AggressiveOptimization)]\n'
    kernels = []
    for name in METHODS:
        header = '    public unsafe static void ' + name + '('
        body, line = method(source, header); prior, _ = method(old, header)
        assert 'float*' in body[:body.index('{')] and body == prior
        assert old.count(attribute + header) == 1
        old = old.replace(attribute + header, header, 1)
        kernels.append(dict(name=name, selected_line=line, lines=body.count('\n') + 1,
            normalized_method_sha256=hashlib.sha256(body.encode()).hexdigest(),
            float_body_exact=True, global_annotation_only=True))
    assert source == old and 'public partial class MathOps' in source
    source = (ROOT / 'src/Lokad.Onnx/TensorOps.MatMul.cs').read_text()
    original, line = method(source, '    static unsafe void RunFloatMatMulKernel(')
    helpers = (OLD / 'source/src/Lokad.Onnx/Zzz.ShortWideMatMul.cs').read_text()
    general, _ = method(helpers, '    static unsafe void RunGeneralFloatMatMulKernel(')
    assert general.replace('RunGeneralFloatMatMulKernel(', 'RunFloatMatMulKernel(', 1) == original
    assert source.count('RunFloatMatMulKernel(') == 5  # declaration and four calls
    short, _ = method(helpers, '    static unsafe void RunShortWidePackedRows(')
    for name in METHODS: assert short.count(name + '(') == 1
    assert helpers.count('[MethodImpl(MethodImplOptions.NoInlining)]') == 2
    result = dict(passed=True, static_only=True, root_product_changed=False, candidate_prepared=False,
        selected_root_inputs=420, m43_source=pin(OLD / 'prepared.json'),
        m43_application=pin(APP / 'closed.json'), rejected_release=pin(GRAPHS / 'closed.json'),
        kernels=kernels, original_general_body_exact_after_rename=True,
        original_general_line=line, original_general_lines=original.count('\n') + 1,
        original_general_call_sites=4, m43_noinline_helpers=2,
        proposed_scope='Keep original general dispatcher and all existing MathOps bodies/flags intact. Redirect four resolved calls to a new guarded dispatcher; only its short-wide branch invokes four internal optimized copies. New short helper retains existing scratch, row groups and odd-row order.',
        limitations='The M45 e5 regression does not identify whether shared annotations, the moved NoInlining general dispatcher, or runtime history caused it. Isolation is a distinct source hypothesis, not transferred admission.',
        source_files={n: pin(ROOT / n) for n in ['src/Lokad.Onnx/MathOps.cs', 'src/Lokad.Onnx/TensorOps.MatMul.cs']},
        generator=pin(Path(__file__)))
    with (OUT / 'source-observations-20260923.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
