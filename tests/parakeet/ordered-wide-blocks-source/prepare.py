"""Draft one isolated ordered-block candidate; do not build or change the root product."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
PARENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-source-v2-20260923'
BASE = ROOT / 'artifacts/parakeet-ordered-wide-blocks-source-20260923'
PLAN = ROOT / '.agent/m55-ordered-wide-blocks-20260923.md'
HELPER = 'src/Lokad.Onnx/Zzz.IsolatedShortMatMul.cs'
ADDED = 'src/Lokad.Onnx/Zzz.OrderedWideMatMul.cs'
PARENT_RECEIPT = '72c22bee93652ed8d6a759c2a984d965fa341697e869cf7a2461a908727483f6'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def panel_method(source, rows):
    old = f'ShortWideMultiply{rows}Rows'
    start = source.index('    [MethodImpl(MethodImplOptions.AggressiveOptimization)]\n'
                         '    internal unsafe static void ' + old + '(')
    tail = source.index('        int rem = K - blocked;', start)
    prefix = source[start:tail]
    row_start = prefix.index(f'            for (int i = 0; i < M; i += {rows})')
    panel_end = prefix.rindex('\n        }')
    row_loop = prefix[row_start:panel_end]
    assert row_loop.count('for (int j = 0; j < N; ++j)') == 1
    row_loop = row_loop.replace('for (int j = 0; j < N; ++j)', 'for (int j = begin; j < end; ++j)')
    if rows == 2:
        assert row_loop.count('float* a0 = A + i * N;') == 1
        assert row_loop.count('float* b = panel;') == 1
        row_loop = row_loop.replace('float* a0 = A + i * N;', 'float* a0 = A + i * N + begin;')
        row_loop = row_loop.replace('float* b = panel;', 'float* b = panel + begin * panelWidth;')
    else:
        # This method indexes A and packed B by j, so their bases must stay unchanged.
        assert rows == 3 and 'var Ap1 = A + i * N;' in row_loop
        assert 'panel + j * (4 * Vector256<float>.Count)' in row_loop
    block = ('            for (int begin = 0; begin < N; begin += 256)\n'
             '            {\n'
             '                int end = Math.Min(N, begin + 256);\n'
             + '\n'.join('    ' + line for line in row_loop.splitlines())
             + '\n            }')
    result = prefix[:row_start] + block + prefix[panel_end:] + '    }\n'
    result = result.replace(old, f'OrderedWideMultiply{rows}Rows')
    result = result.replace('MethodImplOptions.AggressiveOptimization',
                            'MethodImplOptions.NoInlining | MethodImplOptions.AggressiveOptimization')
    comment_start = result.index('        // Kb panels lead')
    comment_end = result.index('        for (int tb', comment_start)
    result = result[:comment_start] + ('        // Visit ordered reduction blocks across all row groups.\n'
                                     '        // Reload each output from its preceding block accumulator.\n') + result[comment_end:]
    # All arithmetic statements and their order remain those of the parent panel.
    fma = lambda value: [line.strip() for line in value.splitlines() if '= Fma.MultiplyAdd(' in line]
    assert fma(result) == fma(prefix) and len(fma(result)) == rows * 4
    assert result.count('{') == result.count('}')
    return result


def transform(source):
    old = '''                    if (threeRows)
                        ShortWideMultiply3Rows(rows, n, k, x, pp, output);
                    else
                        ShortWideMultiply2Rows(rows, n, k, x, pp, output);'''
    assert source.count(old) == 1
    replacement = ('                    if (!TryOrderedWidePackedRows(rows, n, k, x, pp, output))\n'
                   '                    {\n'
                   + '\n'.join('    ' + line for line in old.splitlines())
                   + '\n                    }')
    modified = source.replace(old, replacement)
    header = '''using System;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace Lokad.Onnx;

public partial class MathOps
{
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal unsafe static bool TryOrderedWidePackedRows(int M, int N, int K, float* A, float* P, float* C)
    {
        // Reject unsupported inputs before reading any pointer.
        if (!Fma.IsSupported || M < 64 || (M % 2 != 0 && M % 3 != 0)
            || N < 1024 || K < 1024 || K % 32 != 0 || (long)N * K > 67108864)
            return false;
        if (M % 3 == 0)
            OrderedWideMultiply3Rows(M, N, K, A, P, C);
        else
            OrderedWideMultiply2Rows(M, N, K, A, P, C);
        return true;
    }

'''
    return {HELPER: modified, ADDED: header + panel_method(source, 2) + '\n' + panel_method(source, 3) + '}\n'}


def main():
    assert not BASE.exists()
    assert pin(PARENT / 'prepared.json')['sha256'] == PARENT_RECEIPT
    parent = read(PARENT / 'prepared.json')
    assert parent['passed'] and not parent['root_product_changed'] and len(parent['source']) == 422
    for name, wanted in parent['source'].items():
        path = PARENT / 'source' / name
        assert path.resolve().is_relative_to((PARENT / 'source').resolve())
        assert pin(path) == wanted, name
    assert ADDED not in parent['source']
    original = (PARENT / 'source' / HELPER).read_text(encoding='utf8')
    transformed = transform(original)
    BASE.mkdir()
    for name in parent['source']:
        target = BASE / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARENT / 'source' / name, target)
    for name, value in transformed.items():
        (BASE / 'source' / name).write_text(value, encoding='utf8', newline='\n')
    after = {name: pin(BASE / 'source' / name) for name in [*parent['source'], ADDED]}
    assert len(after) == 423
    assert [name for name, wanted in after.items() if parent['source'].get(name) != wanted] == [HELPER, ADDED]
    before_text = {HELPER: original, ADDED: ''}
    patch = ''.join(''.join(difflib.unified_diff(before_text[name].splitlines(True), value.splitlines(True),
                     fromfile=name, tofile=name)) for name, value in transformed.items())
    (BASE / 'candidate.patch').write_text(patch, encoding='utf8', newline='\n')
    shutil.copy2(PLAN, BASE / 'prospective-plan.md')
    receipt = dict(passed=True, built=False, root_product_changed=False, parent=pin(PARENT / 'prepared.json'),
                   before=parent['source'], source=after, changed=[HELPER, ADDED], block=256,
                   generator=pin(Path(__file__)), patch=pin(BASE / 'candidate.patch'),
                   plan=pin(BASE / 'prospective-plan.md'),
                   scope='One guarded packed-consumer branch; three new internal methods. Original arithmetic, entry, scratch and odd-row paths retained.')
    (BASE / 'prepared.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), source_files=len(after), changed=receipt['changed'],
                          built=False, root_product_changed=False)))


if __name__ == '__main__':
    main()
