"""Record reviewed dispatch/pool structure and exact copied-kernel code reuse."""
import json
import re
from pathlib import Path
from inspect_codegen import ROOT, BASE, OUT, METHODS, parse, pin


def normalized(row):
    labels = dict(re.findall(r'^(G_M\d+_IG\d+):\s+;; offset=(0x[0-9A-Fa-f]+)', row['body'], re.M))
    result = []
    for instruction in row['instructions']:
        instruction = re.sub(r'0x[0-9A-Fa-f]{10,}', '<address>', instruction)
        instruction = re.sub(r'G_M\d+_IG\d+', lambda m: labels[m[0]], instruction)
        result.append(re.sub(r'for IG\d+', 'padding', instruction))
    return result


def main():
    target = Path(__file__).parent / 'codegen-review-20260923.json'; assert not target.exists()
    prior = ROOT / 'tests/parakeet/isolated-short-kernels-results/codegen-review-20260923.json'
    assert pin(prior)['sha256'] == '43e2f5c5d747159b29ca16133d48ca936b598c0c0cd808dc13da904cffd29ccc'
    previous = json.loads(prior.read_text()); assert previous['passed'] and previous['mechanism_admitted']
    for n, v in previous['files'].items(): assert pin(ROOT / n) == v, n
    census = json.loads((OUT / 'census.json').read_text()); structure = json.loads((OUT / 'structure.json').read_text())
    assert census['first_use_passed'] and structure['passed'] and structure['census'] == pin(OUT / 'census.json')
    actual = parse(BASE / 'collected/logs/candidate-codegen-512.stdout')
    oldpath = ROOT / 'artifacts/parakeet-isolated-short-kernels-numerics-amd-20260923/collected/logs/candidate-codegen-512.stdout'
    old = parse(oldpath); comparisons = []
    for name in METHODS.values():
        left = [r for r in old if r['method'].startswith('Lokad.Onnx.MathOps:' + name + '(')]
        right = [r for r in actual if r['method'].startswith('Lokad.Onnx.MathOps:' + name + '(')]
        assert len(left) == len(right) == 1 and left[0]['tier'] == right[0]['tier'] == 'FullOpts'
        assert normalized(left[0]) == normalized(right[0])
        comparisons.append(dict(method=name, old_body=left[0]['index'], new_body=right[0]['index'], bytes=right[0]['bytes'], normalized_instruction_text_equal=True))
    dispatch = next(r for r in actual if ':RunIsolatedShortWideKernel(' in r['method'] and r['tier'] == 'Tier1')
    helper = next(r for r in actual if ':RunIsolatedShortWidePackedRows(' in r['method'] and r['tier'] == 'Tier1')
    assert dispatch['bytes'] == 273 and helper['bytes'] == 2841
    assert 'cmp      edi, 48' in dispatch['body'] and dispatch['body'].count(', 0x400\n') == 2
    assert 'cmp      rax, 0x4000000' in dispatch['body']
    assert sum('tail.jmp' in s for s in dispatch['instructions']) == 2
    assert all(n in dispatch['body'] for n in ['RunIsolatedShortWidePackedRows', 'RunFloatMatMulKernel'])
    assert helper['body'].index('call     G_M000_IG64') < helper['body'].index('G_M000_IG24:')
    assert 'ShortWideMultiply2Rows' in helper['body'] and 'ShortWideMultiply3Rows' in helper['body']
    assert 'vfmadd213ps' in helper['body'] and 'vmulss' in helper['body'] and 'vaddss' in helper['body']
    notes = dict(dispatch='Reviewed all guards and two tail calls; 273-byte Tier1 dispatcher leaves general fallback out of line.',
        helper='Reviewed packed copy, row grouping, calls to copied two/three-row kernels, IG23 call to return finally IG64, then odd original-B remainder IG24-32. Packer/raw remainder inline. AVX512-width moves copy data; accumulation retains AVX2 order.',
        arithmetic='All four FullOpts copied-kernel instruction streams match the previously reviewed M50 copies after address masking/local branch offset resolution. Ordered eight/twelve AVX2 accumulators, no vector stack access in hot reduction loops; masked/scalar tails retain separate multiply/add.',
        limits='Independent diagnostic code capture does not identify versions used in a scored process or establish performance. Shared fallback numerical boundaries and exact compiled bodies/flags passed separately.')
    files = {p.relative_to(ROOT).as_posix(): pin(p) for p in OUT.rglob('*') if p.is_file()}
    for p in [Path(__file__), Path(__file__).parent / 'inspect_codegen.py', Path(__file__).parent / 'review_structure.py', prior, oldpath]:
        files[p.relative_to(ROOT).as_posix()] = pin(p)
    value = dict(passed=True, mechanism_admitted=True, no_performance_measurement=True, closure=pin(BASE / 'closed.json'),
        numerical_values_per_worker=7575601, numerical_groups_per_worker=66, complete_bodies=len(census['bodies']),
        comparisons=comparisons, dispatcher=dict(index=dispatch['index'], bytes=dispatch['bytes']), helper=dict(index=helper['index'], bytes=helper['bytes']), notes=notes, files=files)
    target.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(review=pin(target), complete_bodies=value['complete_bodies'], mechanism_admitted=True, comparisons=comparisons)))


if __name__ == '__main__': main()
