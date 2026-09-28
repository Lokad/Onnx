"""Bind one transpose hypothesis to fresh clocks, actual shapes and pinned source.

Reads closed evidence only. No inference, build, profiling or kernel sweep.
"""
import hashlib
import json
import math
import subprocess
from pathlib import Path

import onnx

from partition import ROOT, TOOLS, pin, read

ORT_REVISION = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
MODEL_SHA = '98a74b21b4cc0017c1e7030319a4a96f4a9506e50f0708f3a516d02a77c96bb1'
FAMILIES = {
    'self_attn/Transpose_3': [0, 2, 3, 1],
    'self_attn/Transpose_5': [0, 2, 3, 1],
    'conv/Transpose': [0, 2, 1],
    'conv/Transpose_1': [0, 2, 1],
}


def main():
    gap = ROOT / 'artifacts/parakeet-attention-owned-gap-20260928'
    proof = read(gap / 'closed.json')
    assert pin(gap / 'closed.json')['sha256'] == 'cc7ebb4355e8885f6d2014df5e4fc5712d767b030346c0feae6d266fad7c9732'
    assert proof['passed'] and proof['analysis'] == pin(gap / 'analysis.json')
    partition = read(gap / 'analysis.json')
    assert partition['passed'] and partition['attribution_only'] and not partition['historical_clocks_used']
    profiles = {}
    for role, folder in [('managed', 'profile'), ('native', 'ort-profile')]:
        base = ROOT / f'artifacts/parakeet-attention-owned-{folder}-amd-20260928'
        assert pin(base / 'closed.json') == partition[role + '_profile']
        closed = read(base / 'closed.json')
        assert closed['passed'] and closed['analysis'] == pin(base / 'analysis.json')
        profiles[role] = read(base / 'analysis.json')
    assert f'git-commit-id={ORT_REVISION[:10]}' in profiles['native']['build_info']
    managed = {r['name']: r for r in profiles['managed']['phases']['wall']['node_rows'] if r['graph'] == 'encoder'}
    native = profiles['native']['profiles']['encoder']
    clocks = {r['name']: r for r in native['node_clocks']}
    group = next(r for r in partition['partition'] if r['group'] == 'Transposes')
    assert len(set(group['managed_members'])) == 244
    assert len(set(group['ort_members'])) == 196
    model_path = ROOT / 'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
    assert pin(model_path)['sha256'] == MODEL_SHA
    model = onnx.load(str(model_path), load_external_data=False)
    nodes = {n.name: n for n in model.graph.node}
    assert len(nodes) == len(model.graph.node)
    records, families = [], []
    for family, permutation in FAMILIES.items():
        rows = []
        for layer in range(24):
            name = f'/layers.{layer}/{family}'
            assert name in group['managed_members'] and name in group['ort_members']
            a, b, node = managed[name], clocks[name], nodes[name]
            assert a['op'] == b['op'] == node.op_type == 'Transpose'
            assert a['calls'] == b['calls'] == 60
            assert list(node.input) == a['inputs'] and list(node.output) == a['outputs']
            attrs = {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
            assert attrs == {'perm': permutation}
            shapes = [r for r in native['shapes'] if r['name'] == name]
            assert sum(s['calls'] for s in shapes) == 80
            collapsed = []
            for shape in shapes:
                assert len(shape['inputs']) == len(shape['outputs']) == 1
                assert set(shape['inputs'][0]) == set(shape['outputs'][0]) == {'float'}
                x, y = shape['inputs'][0]['float'], shape['outputs'][0]['float']
                assert len(x) == len(permutation) and all(d > 0 for d in x)
                assert y == [x[d] for d in permutation]
                # Move axis 1 to the end, with all other axes retaining order:
                # [B,M,*suffix] -> [B,*suffix,M] is exactly B matrix transposes.
                assert permutation == [0, *range(2, len(x)), 1]
                collapsed.append(dict(batch=x[0], rows=x[1], columns=math.prod(x[2:]),
                                      input=x, output=y, calls=shape['calls']))
            row = dict(name=name, family=family, permutation=permutation,
                       measured_calls=60, shapes_including_warmup=collapsed,
                       managed_seconds=a['corpus_seconds'], ort_seconds=b['exclusive_us']/3e6)
            row['excess_seconds'] = row['managed_seconds'] - row['ort_seconds']
            rows.append(row)
        families.append(dict(family=family, nodes=len(rows), permutation=permutation,
                             **{key: sum(r[key] for r in rows) for key in
                                ['managed_seconds', 'ort_seconds', 'excess_seconds']}))
        records.extend(rows)
    assert len(records) == len({r['name'] for r in records}) == 96
    sources = {}
    for rel in ['onnxruntime/core/providers/cpu/tensor/transpose.cc',
                'onnxruntime/core/framework/transpose_helper.cc',
                'onnxruntime/core/mlas/lib/transpose.cpp']:
        data = subprocess.check_output(['git', '-C', str(ROOT/'external/onnxruntime'),
                                        'show', f'{ORT_REVISION}:{rel}'])
        sources[rel] = dict(revision=ORT_REVISION, sha256=hashlib.sha256(data).hexdigest(), size=len(data))
    for rel in ['src/Lokad.Onnx/TensorOps.Shape.cs', 'src/Lokad.Onnx/MathOps.cs']:
        current = (ROOT/rel).read_bytes().replace(b'\r\n', b'\n')
        admitted = subprocess.check_output(['git', '-C', str(ROOT), 'show', f'7e321ecc:{rel}'])
        assert current == admitted
        sources[rel] = dict(revision='7e321ecc09856547880719887f2cb9fb38f4cafb',
                           git_blob_sha256=hashlib.sha256(admitted).hexdigest(), size=len(admitted))
    excess = sum(r['excess_seconds'] for r in families)
    result = dict(passed=True, inference_calls=0, attribution_only=True,
                  native_leaf_sampled=False, route_evidence='source-derived, exact runtime shapes',
                  source=pin(Path(__file__)), partition=pin(gap/'closed.json'),
                  root=partition['root'], application=partition['application'],
                  model=pin(model_path), build_info=profiles['native']['build_info'], sources=sources,
                  complete_transpose_group=group, selected_nodes=96,
                  selected_excess_seconds=excess,
                  fraction_of_transpose_excess=excess/group['excess_seconds'],
                  families=families, records=records,
                  hypothesis='Use the existing 8x8 contiguous-load transpose for the two axis-1-to-last permutations.',
                  prediction_corpus_seconds_saved=0.45,
                  performance_claim=False,
                  acceptance='Independent original application: >=1% gain, <=5% per-clip regression; all 63 controls and 21 gates pass.')
    path = TOOLS/'transpose-breakdown-20260928.json'
    with path.open('x', encoding='utf8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(passed=True, selected_nodes=96, selected_excess_seconds=excess,
                          fraction_of_transpose_excess=result['fraction_of_transpose_excess'], families=families)))


if __name__ == '__main__':
    main()
