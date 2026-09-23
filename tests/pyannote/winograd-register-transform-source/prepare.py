"""Prepare one isolated Winograd reduction change; never mutate the selected product."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/pyannote-winograd-register-transform-source-20260923'
PROOF = ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
PROFILE = ROOT/'artifacts/pyannote-winograd-profile-amd-20260923'
FILE = 'src/Lokad.Onnx/Zzz.ConvBlockedSpatial.Winograd.cs'


def read(path): return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def transform(source):
    start=source.index('    static void TransformWinogradInputContiguous(')
    end=source.rindex('\n}')
    original=source[start:end]
    assert original.count('stackalloc Vector256<float>[16]')==1
    lines=['    static void TransformWinogradInputContiguous(ReadOnlySpan<float> input, Span<float> transformed,',
        '        int c, int h, int w, int top, int left)', '    {',
        '        // The caller proves four valid rows and columns left through left+17.',
        '        // Only two horizontal rows are live; preserve every add/subtract order.',
        '        var order = Vector256.Create(0, 2, 4, 6, 1, 3, 5, 7);',
        '        fixed (float* source = input, destination = transformed)',
        '        for (int ic = 0; ic < c; ic++)', '        {',
        '            float* channel = source + ic * h * w;']
    def row(y):
        lines.append(f'            float* start{y} = channel + (top + {y}) * w + left;')
        for j,offset in enumerate([0,8,2,10]):
            lines.append(f'            var p{y}{j} = Avx2.PermuteVar8x32(Avx.LoadVector256(start{y} + {offset}), order);')
        for label,first,second,control in [('a',0,1,'0x20'),('b',0,1,'0x31'),('d',2,3,'0x20'),('e',2,3,'0x31')]:
            lines.append(f'            var {label}{y} = Avx.Permute2x128(p{y}{first}, p{y}{second}, {control});')
        for x,op,lhs,rhs in [(0,'Subtract','a','d'),(1,'Add','b','d'),(2,'Subtract','d','b'),(3,'Subtract','b','e')]:
            lines.append(f'            var r{y}{x} = Avx.{op}({lhs}{y}, {rhs}{y});')
    def store(output_row,lhs,rhs,op):
        for col in range(4):
            slot=output_row+col
            lines.append(f'            Avx.Store(destination + ({slot} * c + ic) * WinogradBatch, Avx.{op}(r{lhs}{col}, r{rhs}{col}));')
    row(0);row(2);store(0,0,2,'Subtract')
    row(1);store(4,1,2,'Add');store(8,2,1,'Subtract')
    row(3);store(12,1,3,'Subtract')
    lines+=['        }','    }']
    replacement='\n'.join(lines)
    assert replacement.count('Avx.LoadVector256(')==16
    assert replacement.count('Avx2.PermuteVar8x32(')==16
    assert replacement.count('Avx.Permute2x128(')==16
    assert replacement.count('Avx.Subtract(')==24 and replacement.count('Avx.Add(')==8
    assert replacement.count('Avx.Store(')==16 and replacement.count('for (')==1
    assert 'stackalloc' not in replacement and 'Vector512' not in replacement
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
    shutil.copy2(ROOT/'.agent/m37-winograd-register-transform-20260923.md',BASE/'prospective-plan.md')
    value = dict(passed=True,built=False,numerically_qualified=False,root_product_changed=False,
        before=expected,changed=changed,permitted_compiled_change='Lokad.Onnx.ConvBlockedSpatial.TransformWinogradInputContiguous',
        source={p.relative_to(source).as_posix():pin(p) for p in source.rglob('*') if p.is_file()},
        patch=pin(BASE/'candidate.patch'),plan=pin(BASE/'prospective-plan.md'),generator=pin(Path(__file__)),
        root_closure=pin(PROOF/'closed.json'),profile_closure=pin(PROFILE/'closed.json'))
    (BASE/'prepared.json').write_text(json.dumps(value,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),patch=value['patch'],changed=changed,built=False,root_product_changed=False)))


if __name__ == '__main__': main()
