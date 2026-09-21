"""Independent full-array, configuration, timing and owned-process evidence checks."""
from pathlib import Path
import argparse,hashlib,itertools,json,math,statistics,struct,sys
import numpy as np
from protocol import CASES,POLICIES,ROLES,CORE,NATIVE,PROTOCOL,LIMITS,CRITERIA,schedule,specification

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def write(p,v):
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2)

def input_hash(fixture):
    value=bytearray(b'LOKAD-CAMPAIGN-INPUTS-1\0')+struct.pack('<i',3)
    assert sorted(fixture['inputs'])==['attention_mask','input_ids','token_type_ids']
    for name,items in sorted(fixture['inputs'].items()):
        assert len(items)==fixture['shape'][1]
        encoded=name.encode();value+=struct.pack('<i',len(encoded))+encoded+struct.pack('<iiiiq',7,2,1,len(items),len(items))
        value+=struct.pack('<'+'q'*len(items),*items)
    return hashlib.sha256(value).hexdigest()

def rows(v):
    s=v['specification'];freq=v['frequency'];assert type(freq) is int and freq>0
    assert len(v['measured'])==s['blocks']*s['calls']
    assert [(r['block'],r['call']) for r in v['measured']]==list(itertools.product(range(s['blocks']),range(s['calls'])))
    assert (v['first']['block'],v['first']['call'])==(-1,0)
    assert 0<len(v['conditioning'])<=20000
    assert [(r['block'],r['call']) for r in v['conditioning']]==[(-2,i) for i in range(len(v['conditioning']))]
    total=0.;seen=[]
    for r in v['conditioning']:
        assert total<s['conditioning_seconds'] or len(seen)<s['conditioning_minimum'];total+=r['execute']/freq;seen.append(r)
    assert total==v['conditioned'] and total>=s['conditioning_seconds'] and len(seen)>=s['conditioning_minimum']
    assert type(v['conditioning_wall_ticks']) is int and sum(r['request'] for r in v['conditioning'])<=v['conditioning_wall_ticks']<=181*freq
    for r in [v['first']]+v['conditioning']+v['measured']:
        for k in ['block','call','execute','request','bytes','g0','g1','g2']:assert type(r[k]) is int
        assert 0<r['execute']<=r['request']<=300*freq
        assert all(r[k]>=0 for k in ['call','bytes','g0','g1','g2'])

def worker(folder,inputs,model,probe,ort_managed,native_pin,job,phase,smoke=False):
    v=read(folder/'result.json');assert v['specification']==specification(job,phase,smoke);rows(v)
    s=v['specification'];native=job['role']=='N';enabled=phase=='compare' and job['role']=='C'
    assert v['passed'] is True and v['enabled'] is enabled and v['wide_enabled'] is enabled and v['core_sha256']==CORE
    assert v['probe_sha256']==probe['sha256'] and v['ort_managed_sha256']==ort_managed['sha256']
    fixture_path=inputs/(job['case']+'.json');fixture=read(fixture_path)
    assert fixture['name']==job['case'] and v['fixture_sha256']==pin(fixture_path)['sha256']
    assert v['model_sha256']==fixture['model_sha256']==model['sha256']
    assert v['input_sha256']==fixture['input_sha256']==input_hash(fixture)
    assert v['reference_sha256']==fixture['reference_sha256']
    assert v['shape']==fixture['shape']==[1,[8,30,128,128,512][job['case_index']],384]
    assert v['affinity']==4 and v['processor_count']==1
    assert v['flags']==({'LOKAD_ONNX_FINGERPRINT_STRINGS':'1','LOKAD_ONNX_LAYERNORM_WIDE_OUTPUT':'1'} if enabled else {})
    assert v['optimization']==('Memory' if job['policy']=='memory' else 'Speed')
    if not smoke:assert v['runtime']=='10.0.8' and v['avx2'] is True and v['avx512'] is True
    else:assert v['runtime']=='10.0.12'
    for k in ['unchanged_cache','unchanged_inputs','unchanged_held_output']:assert v[k] is True
    assert type(v['load_ticks']) is int and v['load_ticks']>0
    if native:
        n=v['native_identity'];assert n['sha256']==native_pin['sha256'] and n['version']=='1.23.2'
        assert n['threads']==1 and n['sequential'] is True and n['all_optimizations'] is True and n['spinning'] is False
        assert v['fingerprint'] is None
    else:assert v['native_identity'] is None and type(v['fingerprint']) is int
    if enabled:assert v['cache_entries']==2330 and isinstance(v['cache_sha256'],str) and len(v['cache_sha256'])==64
    else:assert v['cache_entries']==0 and v['cache_sha256'] is None
    reference=inputs/fixture['reference_file'];assert pin(reference)['sha256']==fixture['reference_sha256']
    want=np.fromfile(reference,dtype='<f4');assert want.size==math.prod(fixture['shape']) and np.isfinite(want).all()
    for stage in ['before','after']:
        path=folder/(stage+'.f32');actual=np.fromfile(path,dtype='<f4')
        assert actual.size==want.size and np.isfinite(actual).all()
        error=float((np.abs(actual.astype(np.float64)-want.astype(np.float64))/np.maximum(1,np.abs(want.astype(np.float64)))).max(initial=0))
        assert error<=1e-4 and abs(error-v[stage+'_error'])<=1e-15
        assert pin(path)['sha256']==v['output_sha256']
    assert {p.name for p in folder.iterdir()}=={'result.json','before.f32','after.f32'}
    return v

def telemetry(state,samples,jobs,limits=LIMITS):
    assert state['complete'] is True and state['code']==0 and state['limits']==limits
    assert [r['job'] for r in state['runs']]==jobs
    previous=state['started'];births=[state['supervisor']];summary=[]
    for run in state['runs']:
        assert all(math.isfinite(run[k]) for k in ['started','ended','seconds'])
        assert previous<=run['started']<run['ended']<=state['ended'];previous=run['ended']
        assert run['code']==0 and 0<run['seconds']<limits['seconds']
        assert run['members'][str(run['child']['pid'])]==run['child']['birth']
        rows=samples[run['job']['name']];assert len(rows)==run['samples']>0;last=-1.;peak=0;seen={}
        for row in rows:
            assert math.isfinite(row['seconds']) and last<=row['seconds']<=run['seconds'];last=row['seconds']
            assert type(row['available']) is int and row['available']>=limits['available']
            assert len({m['pid'] for m in row['members']})==len(row['members'])
            for m in row['members']:
                assert type(m['pid']) is int and m['pid']>0 and math.isfinite(m['birth']) and m['birth']>=run['child']['birth']
                assert type(m['rss']) is int and m['rss']>=0 and m['affinity']==[2]
                assert seen.get(str(m['pid']),m['birth'])==m['birth'];seen[str(m['pid'])]=m['birth']
            rss=sum(m['rss'] for m in row['members']);assert rss<limits['rss'];peak=max(peak,rss)
        assert peak==run['peak_rss'] and seen==run['members']
        births.extend(dict(pid=int(pid),birth=birth) for pid,birth in seen.items())
        summary.append(dict(job=run['job']['name'],seconds=run['seconds'],samples=len(rows),peak_rss=peak,minimum_available=min(r['available'] for r in rows)))
    return dict(births=births,workers=summary)
