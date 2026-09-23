"""Check every emitted body's boundaries/labels and summarize isolated arithmetic."""
import json
import re
from inspect_codegen import BASE, OUT, METHODS, parse, pin


def main():
    assert not (OUT / 'structure.json').exists()
    census = json.loads((OUT / 'census.json').read_text())
    assert census['first_use_passed']
    bodies = []; kernels = []
    for role in ['current', 'candidate']:
        path = BASE / 'collected/logs' / (role + '-codegen-512.stdout')
        assert pin(path) == census['inputs'][role]
        for row in parse(path):
            definitions = re.findall(r'^(G_M\d+_IG\d+):', row['body'], re.M)
            references = set(re.findall(r'G_M\d+_IG\d+', row['body']))
            assert definitions and len(definitions) == len(set(definitions))
            assert references <= set(definitions), (role, row['index'], references - set(definitions))
            bodies.append(dict(role=role, index=row['index'], labels=len(definitions),
                instructions=len(row['instructions']), passed=True))
            if role == 'candidate' and any(row['method'].startswith('Lokad.Onnx.MathOps:' + n + '(') for n in METHODS.values()):
                assert row['tier'] == 'FullOpts' and '; optimized code' in row['body']
                assert not any('CORINFO_HELP_COUNTPROFILE' in i or 'CORINFO_HELP_PATCHPOINT' in i for i in row['instructions'])
                kernels.append(dict(index=row['index'], method=row['method'], tier=row['tier'], bytes=row['bytes'],
                    fma=[s for s in row['instructions'] if 'vfmadd' in s],
                    separate_multiply_add=[s for s in row['instructions'] if s.startswith(('vmul', 'vadd'))],
                    stack_vector_access=[s for s in row['instructions'] if any(r in s for r in ['xmm', 'ymm', 'zmm']) and any(r in s for r in ['[rbp', '[rsp'])]))
    assert 2 <= len(kernels) <= 4 and len(bodies) == len(census['bodies'])
    result = dict(passed=True, census=pin(OUT / 'census.json'), bodies=bodies, kernels=kernels,
        manual_arithmetic_and_dispatch_review_required=True, no_performance_measurement=True)
    (OUT / 'structure.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(passed=True, bodies=len(bodies), kernels=kernels), indent=2))


if __name__ == '__main__': main()
