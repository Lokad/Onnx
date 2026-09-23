"""Require every finite-extreme geometry, epilogue, comparison and ownership record."""
import re

def census():
    return [(c,m,3,w,s,p,e) for c in [64,80,128,256] for m in [32,48,64,80,128,256] for w in [7,13] for s in [1,2] for p in range(5) for e in range(8)]

def check_result(result,name,spec):
    role,width=name.split('-');lanes=int(width)//32
    assert result['passed'] and result['role']==role and result['core']==spec['products'][role]['Lokad.Onnx.dll']['sha256']
    assert result['probe']==spec['probe']['sha256'] and result['executable']==spec['consumer']['sha256']
    assert result['processor_count']==1 and result['runtime']=='10.0.8' and result['lanes']==lanes and result['avx512']==(lanes==16)
    assert result['flags']==(['DOTNET_EnableAVX512'] if width=='256' else []) and result['no_performance_measurement']
    assert result['cases']==result['controls']==len(result['observations'])==len(result['graph_cases'])==3840
    assert result['candidates']==result['additional_exact_calls']==7680 and result['differences']==0 and result['nan_payload_differences']>=0
    total=payloads=values=0
    for index,(expected,row,graph) in enumerate(zip(census(),result['observations'],result['graph_cases'],strict=True)):
        c,m,h,w,stride,pattern,epi=expected
        assert tuple(row[k] for k in ['c','m','h','w','stride','pattern','epilogue'])==expected and row['index']==index
        assert row['exact_repeat'] and row['owned'] and re.fullmatch('[0-9a-f]{64}',row['sha256'])
        count=m*((h+stride-1)//stride)*((w+stride-1)//stride);assert row['values']==count
        assert all(graph[k]==row[k] for k in ['c','m','h','w','stride','index','values'])
        assert graph['family']=='finite-extremes' and graph['finite_convolution']
        bias,residual,relu=bool(epi&1),bool(epi&2),bool(epi&4)
        assert (graph['bias'],graph['residual'],graph['relu'])==(bias,residual,relu)
        assert graph['retained_weights']==((m+2*lanes-1)//(2*lanes)*2*lanes)*c*9*4
        assert graph['control_scratch']>0 and graph['expected_scratch']==(c*(h+2)*(w+2)+count)*4
        assert [e['repeat'] for e in graph['executions']]==[0,1]
        for e in graph['executions']:
            assert e['scratch_bytes']==graph['expected_scratch'] and e['differences']==0 and e['nan_payload_differences']>=0
            total+=e['differences'];payloads+=e['nan_payload_differences']
        assert graph['nodes']==(['ConvRelu' if relu and not residual else 'Conv']+(['AddRelu' if relu else 'Add'] if residual else []))
        values+=count
    assert total==result['differences'] and payloads==result['nan_payload_differences']
    return dict(passed=True,cases=3840,graph_calls=19200,additional_exact_calls=7680,values=values,nan_payload_differences=payloads,ownership_readonly_scratch=True,no_performance_measurement=True)
