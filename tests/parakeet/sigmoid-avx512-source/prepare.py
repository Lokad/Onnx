"""Prepare one fixed ORT-shaped sigmoid loop without changing release source."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-sigmoid-avx512-source-20260928'
QUALIFIED = ROOT/'artifacts/parakeet-transpose-axis-root-amd-20260928'
DIAGNOSIS = ROOT/'artifacts/parakeet-sigmoid-address-review-20260928'
ORT = ROOT/'artifacts/parakeet-ort-activation-review-20260926'
TARGET = 'src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs'
HELPER = 'src/Lokad.Onnx/Zzz.SigmoidRational.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/SigmoidAvx512Tests.cs'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path): return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def change(name, raw):
    before = raw.decode().replace('\r\n', '\n')
    if name == TARGET:
        old = '                    SigmoidRationalVector(xs, ys);'
        new = '''                    if (opts.Tensor.UseIntrinsics && System.Runtime.Intrinsics.X86.Avx512F.IsSupported && xs.Length >= 32)
                        SigmoidRationalAvx512(xs, ys);
                    else
                        SigmoidRationalVector(xs, ys);'''
        assert before.count(old) == 1
        after = before.replace(old, new)
        assert after.replace(new, old) == before
    else:
        assert name == HELPER and before.endswith('    }\n}\n')
        added = (TOOLS/'Avx512Methods.cs.txt').read_text().replace('\r\n', '\n')
        imports = 'using System.Runtime.InteropServices;\nusing System.Runtime.Intrinsics;\nusing System.Runtime.Intrinsics.X86;\n'
        old = 'using System.Runtime.CompilerServices;\n'
        assert before.count(old) == 1
        after = before.replace(old, old + imports)[:-2] + added + '}\n'
        assert after.replace(imports, '').replace(added, '') == before
    data = (after.replace('\n', '\r\n') if b'\r\n' in raw else after).encode()
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=name, tofile=name))
    return data, patch


def references():
    assert pin(QUALIFIED/'closed.json')['sha256'] == '175693d3952958ba59f4a0785c4e9bd3a74d7e3de9a4ac80809c6077013a0910'
    closure = read(QUALIFIED/'closed.json')
    assert closure['passed'] and closure['files']['bundle/stage.json'] == pin(QUALIFIED/'bundle/stage.json')
    original = {n.removeprefix('source/'): v for n, v in read(QUALIFIED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(original) == 447
    for name, wanted in original.items():
        assert pin(QUALIFIED/'bundle/source'/name) == wanted, name
        # Non-product release documentation has advanced since the root build.
        if name.startswith(('src/', 'tests/')) or name == 'global.json':
            assert pin(ROOT/name) == wanted, name
    assert pin(DIAGNOSIS/'closed.json')['sha256'] == 'a0ae70ee6e91581ad9f606c3862c8dadef3ff251b6e4218c8beceb7ecfee1c1e'
    assert read(DIAGNOSIS/'closed.json')['analysis'] == pin(DIAGNOSIS/'analysis.json')
    analysis = read(DIAGNOSIS/'analysis.json')
    assert analysis['passed'] and analysis['complete_sample_join'] and analysis['counts']['helper_top'] == 2178
    assert pin(ORT/'closed.json')['sha256'] == '41f851c4e473856383c6d51f7b5a0e86d5f0ff8c47947df1f4176b35e2301fc3'
    return original


def main():
    assert not BASE.exists()
    original = references()
    values = {n: (QUALIFIED/'bundle/source'/n).read_bytes() for n in original}
    patches = []
    for name in [TARGET, HELPER]:
        values[name], patch = change(name, values[name]); patches.append(patch)
    assert TEST not in values
    values[TEST] = (TOOLS/'SigmoidAvx512Tests.cs.txt').read_bytes()
    BASE.mkdir()
    for name, data in values.items():
        path = BASE/'source'/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    (BASE/'candidate.patch').write_text(''.join(patches), encoding='utf8')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'PLAN.md').read_bytes())
    source = {n: pin(BASE/'source'/n) for n in values}
    assert set(n for n in original if source[n] != original[n]) == {TARGET, HELPER}
    write(BASE/'prepared.json', dict(passed=True, release_admitted=False, root_product_changed=False,
        baseline=pin(QUALIFIED/'closed.json'), diagnosis=pin(DIAGNOSIS/'closed.json'), ort=pin(ORT/'closed.json'),
        source=source, source_before=original, changed_product_files=[TARGET, HELPER],
        added_product_files=[], changed_methods=['CPUExecutionProvider.Sigmoid'],
        added_methods=['CPUExecutionProvider.SigmoidRationalAvx512', 'CPUExecutionProvider.SigmoidRational512'],
        added_tests=[TEST], source_reversible=True, portable_helper_unchanged=True,
        prediction_corpus_seconds_saved=0.45, patch=pin(BASE/'candidate.patch'), plan=pin(BASE/'prospective-plan.md'),
        tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'), files=len(source), changed=[TARGET, HELPER])))


if __name__ == '__main__': main()
