"""Retain complete M55 bodies and inspect structure before manual mechanism review."""
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/parakeet-ordered-wide-blocks-numerics-amd-20260923'
OUT = ROOT/'artifacts/parakeet-ordered-wide-blocks-codegen-20260923'
PARSER = ROOT/'tests/parakeet/wide-entry-first-use-results/inspect_codegen.py'
spec = importlib.util.spec_from_file_location('retained_codegen_parser', PARSER)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
parse, pin = parser.parse, parser.pin


def normalized(row):
    labels = dict(re.findall(r'^(G_M\d+_IG\d+):\s+;; offset=(0x[0-9A-Fa-f]+)', row['body'], re.M))
    result = []
    for instruction in row['instructions']:
        instruction = re.sub(r'0x[0-9A-Fa-f]{10,}', '<address>', instruction)
        instruction = re.sub(r'G_M\d+_IG\d+', lambda m: labels[m[0]], instruction)
        result.append(re.sub(r'for IG\d+', 'padding', instruction))
    return result


def main():
    assert not OUT.exists()
    closure = json.loads((BASE/'closed.json').read_text())
    assert closure['passed'] and closure['numerically_admitted']
    for name, wanted in closure['files'].items():
        assert pin(BASE/name) == wanted, name
    OUT.mkdir()
    roles, bodies, inputs = {}, [], {}
    for role in ['current', 'candidate']:
        path = BASE/'collected/logs'/(role+'-codegen-512.stdout')
        inputs[role] = pin(path)
        roles[role] = parse(path)
        for row in roles[role]:
            labels = re.findall(r'^(G_M\d+_IG\d+):', row['body'], re.M)
            references = set(re.findall(r'G_M\d+_IG\d+', row['body']))
            assert labels and len(labels) == len(set(labels)) and references <= set(labels)
            (OUT/(role+'-'+str(row['index'])+'.txt')).write_text(row['body']+'\n')
            bodies.append(dict(role=role, **{k: row[k] for k in ['index','method','tier','bytes']},
                labels=len(labels), instructions=len(row['instructions']),
                calls=[s for s in row['instructions'] if s.startswith(('call ', 'tail.jmp'))]))
    comparisons = []
    for name in parser.METHODS.values():
        selected = [[r for r in roles[role] if r['method'].startswith('Lokad.Onnx.MathOps:'+name+'(')]
                    for role in ['current','candidate']]
        assert all(len(rows) == 1 and rows[0]['tier'] == 'FullOpts' for rows in selected)
        left, right = [rows[0] for rows in selected]
        assert normalized(left) == normalized(right), name
        comparisons.append(dict(method=name, current=left['index'], candidate=right['index'],
            bytes=right['bytes'], normalized_instructions_equal=True))
    kernels = []
    for name, count in [('OrderedWideMultiply2Rows',8),('OrderedWideMultiply3Rows',12)]:
        selected = [r for r in roles['candidate'] if r['method'].startswith('Lokad.Onnx.MathOps:'+name+'(')]
        assert len(selected) == 1
        row = selected[0]
        assert row['tier'] == 'FullOpts' and '; optimized code' in row['body']
        assert not any('CORINFO_HELP_COUNTPROFILE' in s or 'CORINFO_HELP_PATCHPOINT' in s for s in row['instructions'])
        fma = [s for s in row['instructions'] if 'vfmadd' in s]
        assert len(fma) == count and all('ymm' in s for s in fma)
        kernels.append(dict(method=name, index=row['index'], bytes=row['bytes'], fma=fma,
            separate_multiply_add=[s for s in row['instructions'] if s.startswith(('vmul','vadd'))],
            stack_vector_access=[s for s in row['instructions'] if any(v in s for v in ['xmm','ymm','zmm'])
                                 and any(v in s for v in ['[rbp','[rsp'])]))
    value = dict(passed=True, closure=pin(BASE/'closed.json'), inputs=inputs,
        parser=pin(PARSER), bodies=bodies, comparisons=comparisons, kernels=kernels,
        manual_mechanism_review_required=True, no_performance_measurement=True)
    (OUT/'census.json').write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(bodies=len(bodies), kernels=kernels, comparisons=comparisons)))


if __name__ == '__main__':
    main()
