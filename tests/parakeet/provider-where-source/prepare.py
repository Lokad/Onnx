"""Freeze a provider-only large uniform Where candidate; preserve generic Tensor.Where."""
import difflib
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PARENT = ROOT / 'artifacts/parakeet-wide-entry-first-use-source-v2-20260923'
CAPTURE = ROOT / 'artifacts/parakeet-scalar-where-layout-amd-20260923'
BASE = ROOT / 'artifacts/parakeet-provider-where-source-20260924'
PLAN = ROOT / '.agent/m57-parakeet-provider-where-20260924.md'
CHANGED = 'src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs'
ADDED = 'src/Lokad.Onnx/Zzz.UniformScalarWhere.cs'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def main():
    assert not BASE.exists()
    assert pin(PARENT / 'prepared.json')['sha256'] == '72c22bee93652ed8d6a759c2a984d965fa341697e869cf7a2461a908727483f6'
    assert pin(CAPTURE / 'closed.json')['sha256'] == 'bc0942f51e27e70f33ae9fdc994c345fe3c620159478ced1a40d5433777f66d8'
    proof = read(CAPTURE / 'closed.json'); assert proof['passed']
    for name, wanted in proof['files'].items():
        assert pin(CAPTURE / name) == wanted, name
    prerequisites = {}
    for folder, digest in [
        ('parakeet-scalar-where-screen-amd-20260924', 'fd8477677a34b9064fa5294d1834b91a59a2284b6bbfbb0ff470c612a2c4949b'),
        ('parakeet-scalar-where-fallback-codegen-amd-20260924', '1cf6a1241cb1501fdf24e7dc049b19181353f3d80c9dfacd71729ece384ffd41')]:
        location = ROOT / 'artifacts' / folder
        assert pin(location / 'closed.json')['sha256'] == digest
        closure = read(location / 'closed.json'); assert closure['passed']
        for name, wanted in closure['files'].items(): assert pin(location / name) == wanted, name
        prerequisites[folder] = pin(location / 'closed.json')
    review = ROOT / 'tests/parakeet/scalar-where-results/fallback-codegen-review-20260924.json'
    assert pin(review)['sha256'] == '4f8b356456d9929db8a69532dfbceff2019044f016d8a111e921995fd58af1b5'
    assert (TOOLS / 'UniformScalarWhere.cs').read_bytes() == (ROOT / 'tests/parakeet/scalar-where-source-v3/UniformScalarWhere.cs').read_bytes()
    parent = read(PARENT / 'prepared.json')
    assert parent['passed'] and len(parent['source']) == 422 and ADDED not in parent['source']
    for name, wanted in parent['source'].items():
        assert pin(PARENT / 'source' / name) == wanted and pin(ROOT / name) == wanted, name
    original = (PARENT / 'source' / CHANGED).read_text(encoding='utf8')
    before = '            case TensorElementType.Float: return Success(op, Tensor<float>.Where((Tensor<bool>)condition, (Tensor<float>)X, (Tensor<float>)Y));'
    after = """            case TensorElementType.Float:
            {
                var c = (Tensor<bool>)condition;
                var x = (Tensor<float>)X;
                var y = (Tensor<float>)Y;
                // Limit the extra scan to substantial scalar-mask selections.
                // Small and unsupported calls retain the generic tensor entry.
                if (y.Length >= 4096 && x.Length == 1)
                {
                    Profiler.StartOpStage(OpStage.ValidateArguments);
                    if (UniformScalarWhere.Try(c, x, y, out var uniform))
                        return Success(op, uniform);
                }
                return Success(op, Tensor<float>.Where(c, x, y));
            }"""
    assert original.count(before) == 1
    transformed = {CHANGED: original.replace(before, after), ADDED: (TOOLS / 'UniformScalarWhere.cs').read_text()}
    BASE.mkdir()
    for name in parent['source']:
        target = BASE / 'source' / name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(PARENT / 'source' / name, target)
    for name, value in transformed.items():
        (BASE / 'source' / name).write_text(value, encoding='utf8', newline='\n')
    sources = {name: pin(BASE / 'source' / name) for name in [*parent['source'], ADDED]}
    assert [name for name, value in sources.items() if parent['source'].get(name) != value] == [CHANGED, ADDED]
    patch = ''.join(''.join(difflib.unified_diff((original if name == CHANGED else '').splitlines(True),
                    value.splitlines(True), fromfile=name, tofile=name)) for name, value in transformed.items())
    (BASE / 'candidate.patch').write_text(patch, encoding='utf8', newline='\n')
    shutil.copy2(PLAN, BASE / 'prospective-plan.md')
    receipt = dict(passed=True, built=False, root_product_changed=False,
        parent=pin(PARENT / 'prepared.json'), capture=pin(CAPTURE / 'closed.json'),
        before=parent['source'], source=sources, changed=[CHANGED, ADDED],
        generator=pin(Path(__file__)), helper=pin(TOOLS / 'UniformScalarWhere.cs'), prerequisites=prerequisites, codegen_review=pin(review),
        patch=pin(BASE / 'candidate.patch'), plan=pin(BASE / 'prospective-plan.md'),
        scope='Only CPUExecutionProvider.Where float arm changes, plus the unchanged V3 helper. Scan only y.Length>=4096 and scalar x. Entire generic Tensor.Where and every other source file remain selected; no existing method flags change.')
    (BASE / 'prepared.json').write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(prepared=pin(BASE / 'prepared.json'), source_files=len(sources), changed=receipt['changed'], built=False)))


if __name__ == '__main__': main()
