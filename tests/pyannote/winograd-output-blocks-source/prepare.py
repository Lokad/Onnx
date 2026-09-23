"""Prepare one isolated Winograd reduction change; never mutate the selected product."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-winograd-output-blocks-source-20260923'
PROOF = ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
PROFILE = ROOT/'artifacts/pyannote-winograd-profile-amd-20260923'
FILE = 'src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Winograd.Kernels.cs'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def transform(source):
    start = source.index('    static void MultiplyWinograd512(')
    end = source.index('    static void OutputWinograd512(',start)
    original = source[start:end]
    tail_start = original.index('            var a0 = Vector512<float>.Zero;')
    tail_end = original.rindex('        }\n    }')
    tail = original[tail_start:tail_end]
    lines = ['    static void MultiplyWinograd512(float* v, float* u, float* p, int c, int m)',
        '    {','        const int lanes = 16;',
        '        for (int k = 0; k < 16; k++)','        {',
        '            int oc = 0;',
        '            for (; oc + 2 * lanes <= m; oc += 2 * lanes)','            {']
    for block in ['a','b']:
        for tile in range(8): lines.append(f'                var {block}{tile} = Vector512<float>.Zero;')
    lines += ['                float* weights = u + k * c * m + oc;',
        '                float* inputs = v + k * c * WinogradBatch;',
        '                for (int ic = 0; ic < c; ic++)','                {',
        '                    var firstWeight = *(Vector512<float>*)weights;',
        '                    var secondWeight = *(Vector512<float>*)(weights + lanes);']
    for tile in range(8):
        lines += [f'                    var input{tile} = Vector512.Create(inputs[{tile}]);',
            f'                    a{tile} = Avx512F.FusedMultiplyAdd(input{tile}, firstWeight, a{tile});',
            f'                    b{tile} = Avx512F.FusedMultiplyAdd(input{tile}, secondWeight, b{tile});']
    lines += ['                    weights += m; inputs += WinogradBatch;',
        '                }',
        '                float* firstOutput = p + (k * m + oc) * WinogradBatch;',
        '                float* secondOutput = firstOutput + lanes * WinogradBatch;']
    for block,pointer in [('a','firstOutput'),('b','secondOutput')]:
        for tile in range(8): lines.append(f'                *(Vector512<float>*)({pointer} + {tile} * lanes) = {block}{tile};')
    lines += ['            }','            for (; oc < m; oc += lanes)','            {']
    lines += ['    '+line for line in tail.splitlines()]
    lines += ['            }','        }','    }']
    replacement = '\n'.join(lines)+'\n'
    assert replacement.count('Avx512F.FusedMultiplyAdd(') == 24
    assert replacement.count('Vector512.Create(inputs[') == 16
    assert source[:start]+original+source[end:] == source
    return source[:start]+replacement+source[end:]


def main():
    assert not BASE.exists()
    for folder,digest in [(PROOF,'62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'),
                          (PROFILE,'8d960999ebda0a6f82f3548c89d4b01ec302ce2802bb9454b3fa371f2a5de126')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        relative = folder == PROFILE
        for name,wanted in proof['files'].items(): assert pin((ROOT if relative else folder)/name) == wanted,name
    expected = {name.removeprefix('source/'):wanted for name,wanted in read(PROOF/'bundle/stage.json')['files'].items() if name.startswith('source/')}
    assert len(expected) == 420
    for name,wanted in expected.items(): assert pin(ROOT/name) == wanted,name
    for short,product in [('Winograd.cs','Zzz.ConvBlockedSpatial.Winograd.cs'),('Winograd.Kernels.cs','Zzz.ConvBlockedSpatial.Winograd.Kernels.cs')]:
        assert pin(ROOT/'tests/pyannote/winograd-range-prototype'/short) == pin(ROOT/'src/Lokad.Onnx'/product)
    before = (ROOT/FILE).read_text(encoding='utf8'); after = transform(before)
    BASE.mkdir(); source = BASE/'source'; source.mkdir()
    for name in expected:
        target = source/name; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,target)
    (source/FILE).write_text(after,encoding='utf8',newline='\n')
    changed = [name for name,wanted in expected.items() if pin(source/name) != wanted]
    assert changed == [FILE]
    for name,wanted in expected.items(): assert pin(ROOT/name) == wanted,name
    patch = ''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=FILE,tofile=FILE))
    (BASE/'candidate.patch').write_text(patch,encoding='utf8',newline='\n')
    shutil.copy2(ROOT/'.agent/m36-winograd-output-blocks-20260923.md',BASE/'prospective-plan.md')
    value = dict(passed=True,built=False,numerically_qualified=False,root_product_changed=False,
        before=expected,changed=changed,permitted_compiled_change='Lokad.Onnx.ConvBlockedSpatial.MultiplyWinograd512',
        source={p.relative_to(source).as_posix():pin(p) for p in source.rglob('*') if p.is_file()},
        patch=pin(BASE/'candidate.patch'),plan=pin(BASE/'prospective-plan.md'),generator=pin(Path(__file__)),
        root_closure=pin(PROOF/'closed.json'),profile_closure=pin(PROFILE/'closed.json'))
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),patch=value['patch'],changed=changed,built=False,root_product_changed=False)))


if __name__ == '__main__': main()
