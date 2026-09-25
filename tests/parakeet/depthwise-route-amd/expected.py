"""Reconcile every depthwise shape and partial panel against the closed census."""
from collections import Counter


def expected(diagnosis):
    result={}
    for s in diagnosis['shapes']:
        if s['attributes']['group'] not in [256,1024]:continue
        x=s['input'];w=s['weight_shape'];y=s['output'];a=s['attributes'];p=s['predicted']
        if len(x)==3:
            x=[x[0],x[1],1,x[2]];w=[w[0],w[1],1,w[2]];y=[y[0],y[1],1,y[2]]
            stride=[1,a['strides'][0]];dilation=[1,a['dilations'][0]];pads=[0,a['pads'][0],0,a['pads'][1]]
        else:stride=a['strides'];dilation=a['dilations'];pads=a['pads']
        g=[*x,y[1],w[2],w[3],*dilation,*stride,*pads,y[2],y[3],a['group']]
        assert len(g)==18
        key=','.join(map(str,g));frequency=4*s['frequency']
        row=result.setdefault(key,dict(geometry=g,calls=0,tiled_batches=0,completed_batches=0,panels=0,products=0,
            views=0,patch_values=0,panel_widths=Counter(),matrix_shapes=Counter(),leaves=Counter(),requested_scratch=p['scratch_requested_bytes']))
        assert row['requested_scratch']==p['scratch_requested_bytes']
        for k in ['calls','tiled_batches','completed_batches']:row[k]+=frequency
        row['panels']+=frequency*p['blocks'];row['products']+=frequency*p['matrix_calls']
        row['views']+=frequency*p['tensor_views'];row['patch_values']+=frequency*p['patch_values_written']
        widths=Counter({str(p['block_columns']):p['blocks']-1});widths[str(p['final_matrix_n'])]+=1
        for width,count in widths.items():
            if count:
                row['panel_widths'][width]+=frequency*count
                row['matrix_shapes']['1,9,'+width]+=frequency*count*a['group']
        row['leaves']['one-row-fma']+=frequency*p['matrix_calls']
    assert len(result)==59 and sum(r['calls'] for r in result.values())==2080
    assert sum(r['products'] for r in result.values())==15794176
    assert sum(r['views'] for r in result.values())==47382528
    return result


def audit_counts(report,diagnosis):
    target=expected(diagnosis)
    assert report['passed'] and report['protocol']=='parakeet-depthwise-route-v1'
    assert report['runtime']=='10.0.8' and report['processor_count']==1 and not report['flags']
    assert report['fma'] and report['avx2'] and report['avx512'] and report['vector_width']==8
    rows={r['key']:r for r in report['rows']};assert len(rows)==len(report['rows']) and rows.keys()==target.keys()
    buckets=[]
    for key,wanted in target.items():
        row=rows[key]
        for field,value in wanted.items():
            if field!='requested_scratch':assert row[field]==value,(key,field,row[field],value)
        assert row['simd'] and row['intrinsics'] and not row['segmented'] and row['degree']==1
        assert row['layouts'] and sum(row['layouts'].values())==row['calls']
        assert all(count>0 for count in row['layouts'].values())
        assert row['scratch_elements'] and sum(row['scratch_elements'].values())==row['tiled_batches']
        sizes=[int(v)*4 for v in row['scratch_elements']]
        assert min(sizes)>=wanted['requested_scratch']
        buckets.append(dict(key=key,requested_bytes=wanted['requested_scratch'],actual_bytes=sorted(sizes),layouts=row['layouts']))
    totals={name:sum(r[name] for r in rows.values()) for name in ['calls','tiled_batches','completed_batches','panels','products','views','patch_values']}
    return dict(passed=True,shapes=len(rows),totals=totals,per_corpus={k:v//4 for k,v in totals.items()},buckets=buckets,
        actual_leaf='one-row-fma',all_shapes_and_partial_panels_exact=True)
