"""Additional finite sharing gates, fixed before AMD inference."""
import re


def validate_sharing(value):
    pair=value['weight_sharing'];assert set(pair)=={'before','after'} and pair['before']==pair['after']
    v=pair['before']
    assert set(v)=={'logical_shared_bytes','shared_arrays','shared_payload_bytes','unique_arrays','unique_payload_bytes','first','past'}
    for key in ['logical_shared_bytes','shared_arrays','shared_payload_bytes','unique_arrays','unique_payload_bytes']:
        assert type(v[key]) is int and v[key]>0,key
    assert v['logical_shared_bytes']==635187200
    assert v['shared_payload_bytes']>=v['logical_shared_bytes']
    assert v['unique_payload_bytes']>=v['shared_payload_bytes'] and v['unique_arrays']>=v['shared_arrays']
    for name in ['first','past']:
        graph=v[name];assert set(graph)=={'initializers','nodes','packed_bytes'}
        assert graph['initializers'] and graph['nodes'] and type(graph['packed_bytes']) is int and graph['packed_bytes']>=0
        names=[]
        for row in graph['initializers']:
            assert set(row)=={'name','tensor_name','type','shape','bytes','sha256'}
            assert isinstance(row['name'],str) and isinstance(row['shape'],list)
            assert type(row['bytes']) is int and row['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',row['sha256'])
            assert all(type(d) is int and d>=0 for d in row['shape'])
            names.append(row['name'])
        assert len(names)==len(set(names))
        for node in graph['nodes']:
            assert set(node)=={'Name','op','Inputs','Outputs'}
            assert isinstance(node['op'],str) and isinstance(node['Inputs'],list) and isinstance(node['Outputs'],list)
    return {k:v[k] for k in ['logical_shared_bytes','shared_arrays','shared_payload_bytes','unique_arrays','unique_payload_bytes']}
