"""Freeze one ORT-derived arithmetic candidate over the qualified current root."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-rational-sigmoid-source-20260927'
PRIOR = ROOT / 'artifacts/parakeet-pad-current-root-amd-20260926'
TARGET = 'src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs'
HELPER = 'src/Lokad.Onnx/Zzz.SigmoidRational.cs'
TEST = 'tests/Lokad.Onnx.Backend.Tests/SigmoidVectorTests.cs'


def pin(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def write(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def changed(original):
    start = original.index('    public static OpResult Sigmoid(')
    end = original.index('    /// <summary>Clip', start)
    before = original[start:end]
    validation = '(options ?? ExecutionOptions.Default).Validated();'
    loop = '                for (int i = 0; i < xs.Length; i++) ys[i] = 1f / (1f + MathF.Exp(-xs[i]));'
    branch = '''                if (xs.Length >= Vector<float>.Count && opts.Tensor.UseSimd
                    && Vector.IsHardwareAccelerated && !x.IsReversedStride)
                {
                    SigmoidRationalVector(xs, ys);
                    return Success(op, y);
                }
'''
    assert before.count(validation) == before.count(loop) == 1
    after = before.replace(validation, 'var opts = ' + validation).replace(loop, branch + loop)
    assert after.replace('var opts = ' + validation, validation).replace(branch, '') == before
    return original[:start] + after + original[end:]


def main():
    assert not BASE.exists()
    assert pin(PRIOR / 'closed.json')['sha256'] == '71c80efd687562cba2e2b5d03e9e036b93de09d0a30f74b976b86355ed12fdf0'
    stage = read(PRIOR / 'bundle/stage.json')
    closure = read(PRIOR / 'closed.json')
    assert closure['files']['bundle/stage.json'] == pin(PRIOR / 'bundle/stage.json')
    assert closure['analysis'] == pin(PRIOR / 'analysis.json')
    assert read(PRIOR / 'analysis.json')['passed']
    values = {}
    for name, wanted in stage['files'].items():
        if not name.startswith('source/'): continue
        path = PRIOR / 'bundle' / name
        assert pin(path) == wanted, name
        relative = name.removeprefix('source/')
        assert pin(ROOT / relative) == wanted, relative
        values[relative] = path.read_bytes()
    assert len(values) == 437
    before = values[TARGET].decode()
    after = changed(before)
    values[TARGET] = after.encode()
    assert HELPER not in values and TEST not in values
    values[HELPER] = (TOOLS / 'Zzz.SigmoidRational.cs.txt').read_bytes()
    tests = (TOOLS.parent / 'vector-sigmoid-source/SigmoidVectorTests.cs.txt').read_text(encoding='utf8')
    marker = '        Assert.Equal(mode == "normal", Vector.IsHardwareAccelerated);'
    assert tests.count(marker) == 1
    values[TEST] = tests.replace(marker, marker + '\n        if (mode == "normal") Assert.Equal(8, Vector<float>.Count);').encode()
    BASE.mkdir()
    for name, data in values.items():
        path = BASE / 'source' / name
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
    patches = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=TARGET, tofile=TARGET))
    patches += ''.join(difflib.unified_diff([], values[HELPER].decode().splitlines(True), fromfile='/dev/null', tofile=HELPER))
    (BASE / 'candidate.patch').write_text(patches, encoding='utf8')
    (BASE / 'prospective-plan.md').write_bytes((ROOT / 'PLAN.md').read_bytes())
    ort = ROOT / 'artifacts/parakeet-ort-activation-review-20260926/closed.json'
    diagnostic = ROOT / 'artifacts/parakeet-sigmoid-execution-diagnostic-amd-20260927/closed.json'
    assert pin(ort)['sha256'] == '41f851c4e473856383c6d51f7b5a0e86d5f0ff8c47947df1f4176b35e2301fc3'
    assert pin(diagnostic)['sha256'] == 'e2ec74acf9b8fe9c36a48226aa478b5efded38071f6b699a04f64410265dd868'
    write(BASE / 'prepared.json', dict(passed=True, root_product_changed=False, release_admitted=False,
        baseline=pin(PRIOR / 'closed.json'), baseline_stage=pin(PRIOR / 'bundle/stage.json'),
        ort=pin(ort), diagnostic=pin(diagnostic), source={n: pin(BASE/'source'/n) for n in values},
        changed_product_files=[TARGET], added_product_files=[HELPER], added_tests=[TEST],
        changed_methods=['CPUExecutionProvider.Sigmoid'], added_methods=['CPUExecutionProvider.SigmoidRationalVector'],
        plan=pin(BASE/'prospective-plan.md'), patch=pin(BASE/'candidate.patch'),
        tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()},
        inherited_tests=pin(TOOLS.parent/'vector-sigmoid-source/SigmoidVectorTests.cs.txt')))
    print(json.dumps(dict(source=pin(BASE/'prepared.json'), files=len(values), changed=[TARGET], added=[HELPER, TEST])))


if __name__ == '__main__': main()
