"""Inspect every emitted body for dense mixed-mask qualification."""
import difflib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'artifacts/parakeet-dense-scalar-where-numerics-amd-20260924'
OUT = ROOT / 'artifacts/parakeet-dense-scalar-where-codegen-20260924'
PARSER = ROOT / 'tests/parakeet/wide-entry-first-use-results/inspect_codegen.py'
spec = importlib.util.spec_from_file_location('parser', PARSER)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)
pin = parser.pin


def normalized(instructions):
    return [re.sub(r'0x[0-9A-Fa-f]{10,}', '<address>', i) for i in instructions]


def main():
    assert not OUT.exists()
    assert pin(BASE / 'closed.json')['sha256'] == 'edd4cf6e9bc9eccb1154df66f9e26419f0c28ebe789962d1d2618d85c1802832'
    proof = json.loads((BASE / 'closed.json').read_text())
    assert proof['passed']
    for name, wanted in proof['files'].items():
        assert pin(BASE / name) == wanted, name
    OUT.mkdir()
    bodies = []; roles = {}; entries = {}; providers = {}; inputs = {}
    for role, count in [('current', 29), ('candidate', 27)]:
        path = BASE / 'collected/logs' / (role + '-codegen-512.stdout')
        inputs[role] = pin(path); rows = parser.parse(path); roles[role] = rows
        assert len(rows) == count
        for row in rows:
            labels = re.findall(r'^(G_M\d+_IG\d+):', row['body'], re.M)
            assert len(labels) == len(set(labels)) and set(re.findall(r'G_M\d+_IG\d+', row['body'])) <= set(labels)
            target = OUT / f'{role}-{row["index"]}.txt'
            target.write_text(row['body'] + '\n')
            bodies.append(dict(role=role, **{k:row[k] for k in ['index','method','tier','bytes']},
                file=target.relative_to(ROOT).as_posix(), identity=pin(target), labels=len(labels),
                instructions=len(row['instructions']), calls=[i for i in row['instructions'] if i.startswith(('call ', 'tail.jmp'))]))
        full = [r for r in rows if '[float]:Where(' in r['method'] and r['tier'] == 'Tier1']
        assert len(full) == 1
        entry = full[0]; ins = entry['instructions']
        assert not any('DenseScalarWhere' in i or ':BroadcastShape(' in i for i in ins)
        assert sum(i.startswith(('idiv ', 'div ')) for i in ins) == 4
        assert '; 23 inlinees with PGO data; 24 single block inlinees; 8 inlinees without PGO data' in entry['body']
        entries[role] = dict(index=entry['index'], bytes=entry['bytes'], external_shape_calls=0, dense_helper_calls=0,
            divisions=4, profile=[s for s in entry['body'].splitlines() if 'PGO' in s or 'inlinee' in s])
        provider = [r for r in rows if r['method'].startswith('Lokad.Onnx.CPUExecutionProvider:Where')]
        providers[role] = [{k:r[k] for k in ['index','tier','bytes']} for r in provider]
        tier0 = next(r for r in provider if r['tier'] == 'Tier0')
        for dtype in ['bool','byte','int','long','uint','ulong','float','double','System.Half']:
            assert sum(f'Lokad.Onnx.Tensor`1[{dtype}]:Where(' in i for i in tier0['instructions']) == 1
        if role == 'candidate':
            ins = tier0['instructions']; index = next(i for i,s in enumerate(ins) if 'DenseScalarWhere:Try' in s)
            before = ins[index-15:index]; after = ins[index+1:index+20]
            assert 'cmp      rax, 0x1000' in before and 'cmp      rax, 1' in before
            assert all(any(s.startswith(branch) and 'G_M000_IG18' in s for s in before) for branch in ['jl ', 'jne '])
            assert after[0] == 'test     eax, eax' and after[1] == 'je       SHORT G_M000_IG18'
            assert sum('OpResult:Success' in i for i in tier0['instructions']) == 10
        consumers = [r for r in rows if r['method'].startswith('Program:Invoke[')]
        assert {r['method'].split('[')[1].split(']')[0] for r in consumers} == {'long','float','bool','byte','int','uint','ulong','double','System.Half'}
        for consumer in consumers:
            assert any('CPUExecutionProvider:Where' in i for i in consumer['instructions'])
            assert not any('DenseScalarWhere' in i for i in consumer['instructions'])
    helpers = {}
    for method, size in [('Try', 1699), ('SelectMixed', 1155)]:
        candidates = [r for r in roles['candidate'] if 'DenseScalarWhere:'+method+'[' in r['method']]
        assert len(candidates) == 1
        helper = candidates[0]; ins = helper['instructions']
        assert helper['tier'] == 'FullOpts' and helper['bytes'] == size
        assert not any(i.startswith(('idiv ', 'div ', 'vadd', 'vmul', 'vfmadd', 'vsub', 'vdiv')) for i in ins)
        assert not any(':GetValue(' in i or ':Broadcast' in i for i in ins)
        for token in ['SpanHelpers:Fill[float]', 'SpanHelpers:Memmove']:
            assert sum(token in i for i in ins) == 1
        if method == 'Try':
            for token in ['SpanHelpers+Negate`1[byte]', 'SpanHelpers+DontNegate`1[byte]', 'DenseScalarWhere:SelectMixed[float]']:
                assert sum(token in i for i in ins) == 1
            # Null/layout/rank/shape refusals precede the first mask scan and allocation.
            first_call = next(i for i,s in enumerate(ins) if s.startswith('call '))
            assert 'SpanHelpers+Negate`1[byte]' in ins[first_call]
            assert all(any(s.startswith(branch) and 'G_M000_IG23' in s for s in ins[:first_call]) for branch in ['je ', 'jne ', 'jle ', 'jg '])
            assert any('cmp      qword ptr [r14+0x30], 0' == s for s in ins[:first_call])
        else:
            assert sum('SpanHelpers:ClearWithoutReferences' in i for i in ins) == 1
            assert not any('CORINFO_HELP_NEW' in i for i in ins)
            # Contiguous selection uses a raw zero check and float loads/stores only.
            assert 'cmp      byte  ptr [r11+r12], 0' in ins
            assert 'vmovss   xmm0, dword ptr [r10+4*r12]' in ins
            assert 'vmovss   dword ptr [rax], xmm0' in ins
            assert 'inc      dword ptr [r11]' in ins
            assert 'sub      r8d, r11d' in ins
        helpers[method] = dict(index=helper['index'], bytes=size, tier=helper['tier'], divisions=0,
            no_float_arithmetic=True, no_getvalue_or_broadcast_calls=True,
            calls=[i for i in ins if i.startswith(('call ', 'tail.jmp'))])
    full = {role:next(r for r in rows if '[float]:Where(' in r['method'] and r['tier']=='Tier1') for role,rows in roles.items()}
    diff = list(difflib.unified_diff(normalized(full['current']['instructions']), normalized(full['candidate']['instructions']), n=3))
    result = dict(passed=True, ready_for_prospective_stability_control=True, no_performance_measurement=True,
        closure=pin(BASE/'closed.json'), inputs=inputs, parser=pin(PARSER), reviewer=pin(Path(__file__)), bodies=bodies,
        whole_float_entries=entries, provider_entries=providers, helpers=helpers, generic_native_diff=diff,
        limitation='Candidate provider emitted Tier0 only; selected also emitted instrumented Tier0 and Tier1. These diagnostics do not establish tiers in a future timed worker.',
        conclusion='Both generic float Tier1 entries retain 23 profiled inlinees, zero external BroadcastShape calls and four divisions. Their native layouts differ. The provider preserves its guards and nine dtype fallbacks. Two new FullOpts helpers select raw zero/nonzero masks into owned outputs; the mixed body has no allocation, division, broadcast/GetValue call or float arithmetic. Numerical qualification and body inspection permit prospective measurement preparation, not a performance claim.')
    target = Path(__file__).parent / 'codegen-review-20260924.json'
    assert not target.exists(); target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(review=pin(target), bodies=len(bodies), entries=entries, providers=providers, helpers=helpers)))


if __name__ == '__main__':
    main()
