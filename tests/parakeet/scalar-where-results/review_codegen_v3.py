"""Record complete emitted bodies and the uniform-mask mechanism before timing."""
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-scalar-where-numerics-amd-v3-20260924'
OUT = ROOT / 'artifacts/parakeet-scalar-where-codegen-v3-20260924'
PARSER = ROOT / 'tests/parakeet/wide-entry-first-use-results/inspect_codegen.py'
spec = importlib.util.spec_from_file_location('retained_parser', PARSER)
parser = importlib.util.module_from_spec(spec); spec.loader.exec_module(parser)
pin = parser.pin


def main():
    assert not OUT.exists()
    assert pin(BASE / 'closed.json')['sha256'] == '6493136351aa15a5b96b62e98a557d2fcc4f09191712bccf056e6a827e8705dc'
    proof = json.loads((BASE / 'closed.json').read_text()); assert proof['passed']
    for name, wanted in proof['files'].items(): assert pin(BASE / name) == wanted, name
    OUT.mkdir(); roles = {}; bodies = []; inputs = {}
    for role in ['current', 'candidate']:
        path = BASE / 'collected/logs' / (role + '-codegen-512.stdout')
        inputs[role] = pin(path); roles[role] = parser.parse(path)
        for row in roles[role]:
            labels = re.findall(r'^(G_M\d+_IG\d+):', row['body'], re.M)
            assert len(labels) == len(set(labels)) and set(re.findall(r'G_M\d+_IG\d+', row['body'])) <= set(labels)
            (OUT / f'{role}-{row["index"]}.txt').write_text(row['body'] + '\n', encoding='utf8')
            bodies.append(dict(role=role, **{k: row[k] for k in ['index', 'method', 'tier', 'bytes']},
                labels=len(labels), instructions=len(row['instructions']),
                calls=[s for s in row['instructions'] if s.startswith(('call ', 'tail.jmp'))]))
    assert len(roles['current']) == 5 and len(roles['candidate']) == 4
    helper = next(r for r in roles['candidate'] if r['method'].startswith('Lokad.Onnx.UniformScalarWhere:Try[float]'))
    assert helper['tier'] == 'FullOpts' and helper['bytes'] == 1277
    ins = helper['instructions']; body = helper['body']
    assert not any(s.startswith(('idiv ', 'div ', 'vadd', 'vmul', 'vfmadd', 'vsub', 'vdiv')) for s in ins)
    for text in ['SpanHelpers+Negate`1[byte]', 'SpanHelpers+DontNegate`1[byte]', 'SpanHelpers:Fill[float]', 'SpanHelpers:Memmove']:
        assert sum(text in s for s in ins) == 1
    assert body.index('G_M000_IG19:') < body.index('G_M000_IG23:')
    assert 'je       G_M000_IG37' in body.split('G_M000_IG19:')[1].split('G_M000_IG20:')[0]
    scan = body.split('G_M000_IG23:')[1].split('G_M000_IG25:')[0]
    assert 'cmp      byte  ptr [rdi], 0' in scan and 'setne    cl' in scan
    assert 'jne      G_M000_IG26' in scan and 'jge      G_M000_IG30' in scan
    assert 'SpanHelpers+Negate`1[byte]' in scan and 'xor      esi, esi' in scan
    true_scan=body.split('G_M000_IG26:')[1].split('G_M000_IG27:')[0]
    assert 'SpanHelpers+DontNegate`1[byte]' in true_scan and 'xor      esi, esi' in true_scan
    assert 'jl       G_M000_IG25' in true_scan and 'jmp      SHORT G_M000_IG30' in true_scan
    assert 'CORINFO_HELP_NEWARR_1_VC' in body.split('G_M000_IG25:')[1].split('G_M000_IG26:')[0]
    entry = next(r for r in roles['candidate'] if '[float]:Where(' in r['method'] and r['tier'] == 'Instrumented Tier0')
    assert entry['body'].index('Profiler:StartOpStage') < entry['body'].index('UniformScalarWhere:Try[float]')
    assert 'je       SHORT G_M000_IG04' in entry['body'].split('G_M000_IG02:')[1].split('G_M000_IG03:')[0]
    assert 'ret ' in entry['body'].split('G_M000_IG03:')[1].split('G_M000_IG04:')[0]
    assert ':Broadcast(' in entry['body'].split('G_M000_IG04:')[1]
    fallback = next(r for r in roles['candidate'] if r['tier'] == 'Tier1-OSR')
    assert any(s.startswith('idiv ') for s in fallback['instructions'])
    value = dict(passed=True, mechanism_verified=True, performance_admitted=False, ready_for_fixed_screen=True,
        no_performance_measurement=True, closure=pin(BASE / 'closed.json'), inputs=inputs,
        parser=pin(PARSER), reviewer=pin(Path(__file__)), bodies=bodies,
        helper=dict(bytes=1277, tier='FullOpts', no_float_arithmetic=True, no_coordinate_division=True),
        notes=dict(validation='IG02-15 refuse nulls, non-exact classes, reversed layout, non-scalar x and unsupported ranks. IG27-29 validate right-aligned condition dimensions. IG19 takes the empty allocation path before any mask read.',
            mechanism='IG23 normalizes the first byte with comparison against zero. A zero-leading mask reaches IG24, which searches for any nonzero byte (Negate). A nonzero-leading mask reaches IG26, which searches for zero (DontNegate). Both refuse on a mixed mask; IG25 allocates independent storage after a uniform result. IG50 loads scalar bits and calls Fill; IG63 calls Memmove. No exact-byte-one assumption remains.',
            fallback='The float Tier0 entry calls StartOpStage, then the helper. Success returns its output; refusal reaches the original Broadcast sequence. The OSR body retains coordinate division and direct selected float loads/stores. The separate build audit proves exact original IL, locals and branch targets.',
            limits='Runtime scan/fill/memmove helper bodies were not captured. Current and candidate tier inventories differ; this diagnostic does not prove which tier a timed process uses or native-code identity of the fallback. No performance result exists.',
            pending='V2 counterexample e83bf8c6 is resolved by zero/nonzero scans. V3 numerical closure64931363 adds all eight raw masks to the original123cases, including independent NumPy bit checks and exact selected results in both modes. Fixed component and complete application gates remain pending.'),
        files={p.relative_to(ROOT).as_posix(): pin(p) for p in OUT.iterdir() if p.is_file()})
    target = Path(__file__).parent / 'codegen-review-v3-20260924.json'; assert not target.exists()
    target.write_text(json.dumps(value, indent=2) + '\n', encoding='utf8')
    print(json.dumps(dict(review=pin(target), bodies=len(bodies), helper=value['helper'], performance_admitted=False)))


if __name__ == '__main__': main()
