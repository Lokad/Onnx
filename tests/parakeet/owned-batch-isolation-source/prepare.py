"""Move one packed-weight decision out of the shared batched dispatcher."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-owned-batch-isolation-source-20260925'
PRIOR = ROOT/'artifacts/parakeet-direct-depthwise-source-v2-20260925'
TARGET = 'src/Lokad.Onnx/TensorOps.MatMul.cs'
INSERTION = '        if (TryRunOwnedPackedBatches(bx, by, z, options)) return;\n'


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    assert not BASE.exists()
    assert pin(PRIOR/'prepared.json')['sha256'] == '3b82260e7baca95e13a6e28ea4828455555c26061f4b05764bb0d91e72533b1c'
    previous = json.loads((PRIOR/'prepared.json').read_text())
    assert len(previous['source']) == 435
    values = {}
    for name, wanted in previous['source'].items():
        path = PRIOR/'source'/name
        assert pin(path) == wanted, name
        values[name] = path.read_bytes()
    before = values[TARGET].decode()
    assert before.count(INSERTION) == 1
    release = before.replace(INSERTION, '')
    assert release == (ROOT/TARGET).read_text(encoding='utf8')
    after = release
    replacements = []
    for indent, output in [('        ', 'target'), ('            ', 'z')]:
        old = f'{indent}RunBatchedFloatMatMul(bx, by, {output}, options);\n'
        new = f'{indent}if (!TryRunOwnedPackedBatches(bx, by, {output}, options))\n    {old}'
        assert after.count(old) == 1
        after = after.replace(old, new)
        replacements.append((old, new))
    reversed_text = after
    for old, new in replacements:
        reversed_text = reversed_text.replace(new, old)
    signature = '    static void RunBatchedFloatMatMul(Tensor<float> bx, Tensor<float> by, Tensor<float> z, TensorExecutionOptions options)\n    {\n'
    assert reversed_text.count(signature) == 1
    assert reversed_text.replace(signature, signature+INSERTION) == before
    values[TARGET] = after.encode()
    BASE.mkdir()
    for name, data in values.items():
        path = BASE/'source'/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile=TARGET, tofile=TARGET))
    (BASE/'candidate.patch').write_text(patch, encoding='utf8')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'.agent/e5-owned-batch-isolation-20260925.md').read_bytes())
    source = {name: pin(BASE/'source'/name) for name in values}
    assert [n for n, v in previous['source'].items() if source[n] != v] == [TARGET]
    result = dict(passed=True, root_product_changed=False, release_admitted=False,
                  baseline=pin(PRIOR/'prepared.json'), source=source,
                  changed_product_files=[TARGET], changed_methods=['RunBatchedFloatMatMul', 'MatMulInto', 'MatMul'],
                  added_product_files=[], added_tests=[], source_reversible=True,
                  shared_dispatcher_matches_release=True,
                  diagnosis=pin(ROOT/'tests/benchmarks/e5-direct-code-results/inspection-20260925.json'),
                  plan=pin(BASE/'prospective-plan.md'), patch=pin(BASE/'candidate.patch'),
                  tools={'prepare.py': pin(Path(__file__))})
    with (BASE/'prepared.json').open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(prepared=pin(BASE/'prepared.json'), changed_files=[TARGET], files=len(source))))


if __name__ == '__main__':
    main()
