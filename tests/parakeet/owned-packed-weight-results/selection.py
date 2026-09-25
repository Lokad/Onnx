"""Join the actual loaded census with retained full-graph metadata."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
sys.path.insert(0,str(TOOLS.parent/'owned-packed-weight-selection-v2'))
from review import collected
from run import BASE,pin,read,write


def main():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    observation=read(folder/'probe/512/selection.json');result=read(folder/'probe/512/result.json')
    assert result['passed'] and result['diagnostic_only'] and result['no_inference'] and result['mode']=='512'
    assert result['runtime']=='.NET 10.0.8' and result['affinity']==4 and result['processor_count']==1
    assert result['core_sha256']==spec['product']['Lokad.Onnx.dll']['sha256']
    assert result['data_sha256']==spec['product']['Lokad.Onnx.Data.dll']['sha256']
    assert result['runner_sha256']==built['runtime']['OwnedWeightCensus.dll']['sha256']
    assert result['spec_sha256']==pin(folder/'spec.json')['sha256']
    expected={r['Name']:r for r in spec['weights']}
    assert len(observation['rows'])==len(expected)==96 and {r['name'] for r in observation['rows']}==set(expected)
    assert observation['owned_count']==88 and observation['owned_bytes']==88*16777216
    assert observation['initializer_count']==649 and observation['map_count']==37
    assert not observation['captured_count'] and not observation['nested'] and not observation['rejected'] and not observation['folds']
    for row in observation['rows']:
        wanted=expected[row['name']];assert row['dimensions']==wanted['Shape'] and not row['folded']
        assert row['kind']==('Lokad.Onnx.DenseTensor`1[System.Single]' if wanted['Cached'] else 'Lokad.Onnx.OwnedPackedTensor')
        consumer,=row['consumers']
        assert consumer['op']=='MatMul' and consumer['position']==1 and len(consumer['inputs'])==2 and '/feed_forward' in consumer['Name']
        assert consumer['inputs'][1]==row['name'] and consumer['Domain'] in ['', 'ai.onnx']
        if wanted['Cached']:assert row['layout']==dict(IsReversedStride=False,Offset=0,Count=4194304,length=4194304,aliases=1,mapped=True)
        else:assert row['layout'] is None
    assert sum(r['kind']=='Lokad.Onnx.OwnedPackedTensor' for r in observation['rows'])==87
    metadata=ROOT/'artifacts/parakeet-feed-forward-cost-amd-20260925/capture-collected/clock/graphs.json'
    assert pin(metadata)['sha256']=='34d69c2f286d557e4f5dbc9b7305248bfc7890fded30f9dbbbf2e38aed0d0b08'
    graph=read(metadata)['encoder-model.onnx'];matching=[]
    for node in graph['nodes']:
        if node['op']=='MatMul' and len(node['inputs'])==2 and len(node['constant_inputs'])>1:
            weight=node['constant_inputs'][1]
            if weight and weight['type']=='Float' and weight['dims'] in [[1024,4096],[4096,1024]]:matching.append(node)
    assert len(matching)==97
    extra,=[n for n in matching if n['inputs'][1] not in expected]
    assert extra['name']=='/pre_encode/out/MatMul' and extra['inputs'][1]=='onnx::MatMul_6382'
    analysis=dict(passed=True,diagnostic_only=True,release_admitted=False,no_inference=True,
        product=spec['product'],selection=observation,result=result,resources=resources,
        target_count=87,actual_total_count=88,extra_count=1,extra_projection=extra,
        extra_name_inferred_from_exhaustive_matching_graph=True,graph_metadata=pin(metadata),
        original_failed_census=pin(ROOT/'artifacts/parakeet-owned-packed-weight-census-amd-20260925/capture-collected/capture-collection.json'),
        failed_selection_build=pin(ROOT/'artifacts/parakeet-owned-packed-weight-selection-amd-20260925/build-collected/build-collection.json'),
        next_change='Restrict owned preparation to the measured feed-forward consumer names; preserve preprocessing weight storage and all arithmetic.')
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),collection=pin(folder/'capture-collection.json'),
        transfer=pin(BASE/'capture-transfer.json'),terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    write(TOOLS/'selection-20260925.json',dict(**analysis,closure=pin(BASE/'closed.json'),publisher=pin(Path(__file__))))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),selected=88,intended=87,extra_projection=extra['name'],resources=resources)))


if __name__=='__main__':main()
