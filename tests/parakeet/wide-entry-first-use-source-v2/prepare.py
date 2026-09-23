"""Create an isolated first-use optimized wide entry without touching root product."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-wide-entry-first-use-source-v2-20260923'
PARENT = ROOT / 'artifacts/parakeet-wide-projection-isolation-source-20260923'
DIAG = ROOT / 'artifacts/parakeet-wide-runtime-diagnostic-amd-20260923'
MATRIX = 'src/Lokad.Onnx/TensorOps.MatMul.cs'
KERNELS = 'src/Lokad.Onnx/Zzz.IsolatedShortMatMul.cs'
ENTRY = 'src/Lokad.Onnx/Zzz.WideProjectionEntry.cs'
OLD = 'MatMul2DCore'
DISPATCH = 'DispatchWideProjectionMatMul2DCore'
CLONE = 'RunWideProjectionMatMul2DCore'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as f:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())


def method(text):
    start = text.index('    static Tensor<float> MatMul2DCore(')
    opening = text.index('{', start); depth = 1; i = opening + 1
    while depth:
        depth += (text[i] == '{') - (text[i] == '}'); i += 1
    return text[start:i]


def transform(matrix, kernels):
    body = method(matrix)
    assert matrix.count(OLD + '(') == 5 and body.count(OLD + '(') == 1
    modified = matrix.replace(OLD + '(', DISPATCH + '(')
    modified = modified.replace('static Tensor<float> ' + DISPATCH + '(', 'static Tensor<float> ' + OLD + '(')
    assert method(modified) == body and modified.count(DISPATCH + '(') == 4
    flag = '[MethodImpl(MethodImplOptions.NoInlining)]\n    static unsafe void RunIsolatedShortWidePackedRows('
    assert kernels.count(flag) == 1
    optimized = kernels.replace(flag, flag.replace('MethodImplOptions.NoInlining',
        'MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization'))
    header = '''using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics.X86;
using System.Threading.Tasks;
using static Lokad.Onnx.Profiler;

namespace Lokad.Onnx;

public abstract partial class Tensor<T> where T : unmanaged
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static Tensor<float> DispatchWideProjectionMatMul2DCore(Tensor<float> x, Tensor<float> y, DenseTensor<float> destination, TensorExecutionOptions options, bool clearDestination)
    {
        // Preserve the original validation path for malformed or smaller inputs.
        if (x is not null && y is not null && x.Rank == 2 && y.Rank == 2
            && x.Dimensions[0] >= 48 && x.Dimensions[1] >= 1024 && y.Dimensions[1] >= 1024
            && (long)x.Dimensions[1] * y.Dimensions[1] <= 67108864
            && options.UseSimd && options.UseIntrinsics && Fma.IsSupported)
            return RunWideProjectionMatMul2DCore(x, y, destination, options, clearDestination);
        return MatMul2DCore(x, y, destination, options, clearDestination);
    }

    [MethodImpl(MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization)]
'''
    entry = header + body.replace(OLD + '(', CLONE + '(') + '\n}\n'
    assert body.replace(OLD, CLONE) in entry
    return {MATRIX: modified, KERNELS: optimized, ENTRY: entry}


def main():
    assert not BASE.exists()
    assert pin(PARENT / 'prepared.json')['sha256'] == '2714b31148e581466fad1802f1d13560d860950a39481eee40eb59100b8550ee'
    prior = read(PARENT / 'prepared.json')
    assert prior['passed'] and not prior['root_product_changed'] and len(prior['source']) == 421
    assert len(prior['before']) == 420
    for n, v in prior['source'].items(): assert pin(PARENT / 'source' / n) == v, n
    for n, v in prior['before'].items(): assert pin(ROOT / n) == v, n
    assert pin(DIAG / 'closed.json')['sha256'] == 'fee8b6d9a20917ca24a03300f35720d644b7841ba2ad5408f71392f11bf7642c'
    diagnostic = read(DIAG / 'closed.json'); assert diagnostic['passed'] and diagnostic['diagnostic_only']
    for n, v in diagnostic['files'].items(): assert pin(DIAG / n) == v, n
    old = {n: (PARENT / 'source' / n).read_text(encoding='utf8') for n in [MATRIX, KERNELS]}
    new = transform(old[MATRIX], old[KERNELS]); old[ENTRY] = ''
    BASE.mkdir(); source = BASE / 'source'; source.mkdir()
    for n in prior['source']:
        target = source / n; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARENT / 'source' / n, target)
    for n, value in new.items(): (source / n).write_text(value, encoding='utf8', newline='\n')
    assert [n for n, v in prior['source'].items() if pin(source / n) != v] == [MATRIX, KERNELS]
    (BASE / 'candidate.patch').write_text(''.join(''.join(difflib.unified_diff(old[n].splitlines(True), new[n].splitlines(True), fromfile=n, tofile=n)) for n in new), encoding='utf8')
    shutil.copy2(ROOT / '.agent/m54-wide-projection-first-use-entry-20260923.md', BASE / 'prospective-plan.md')
    shutil.copy2(PARENT / 'census.json', BASE / 'census.json')
    result = dict(passed=True, built=False, root_product_changed=False, before=prior['before'],
        source={n: pin(source / n) for n in [*prior['source'], ENTRY]}, delta_from_parent=list(new),
        parent=pin(PARENT / 'prepared.json'), diagnostic=pin(DIAG / 'closed.json'), generator=pin(Path(__file__)),
        patch=pin(BASE / 'candidate.patch'), census=pin(BASE / 'census.json'), plan=pin(BASE / 'prospective-plan.md'),
        scope='Four entry caller operands; new guarded dispatcher and exact private entry clone; existing isolated helper flag8to520 only. Original entry, shared bodies and flags unchanged.')
    (BASE / 'prepared.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), source_files=len(result['source']), delta=result['delta_from_parent'])))


if __name__ == '__main__': main()
