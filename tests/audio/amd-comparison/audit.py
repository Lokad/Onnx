"""Audit complete AMD benchmark requests, coverage, identities and resource samples."""
from pathlib import Path
import copy, hashlib, json, math, statistics
import numpy as np
from deploy import BASE,ROOT
from protocol import pin,read,write,LIMITS,FAMILIES,schedule,check_sample,validate_records


def worker_identity(value,manifest,frozen,base,engine):
    assert value['engine']==engine and value['manifest_sha256']==pin(base/'manifests'/(manifest['family']+'.json'))['sha256']
    if engine=='managed':
        runner='WhisperBenchmark.dll' if manifest['family']=='whisper' else 'AudioBenchmark.dll'
        assert value['runtime']=='.NET 10.0.8' and value['processor_count']==1 and value['flags']=={}
        assert value['runner_sha256']==frozen['files']['bin/'+runner]['sha256']
        for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll')]:
            assert value[key]==manifest[key]==frozen['files']['bin/'+name]['sha256']
        for row in value['records']:
            assert type(row['allocated_bytes']) is int and row['allocated_bytes']>=0
            assert len(row['gc_before'])==len(row['gc_after'])==3
            assert all(type(a) is int and type(b) is int and 0<=a<=b for a,b in zip(row['gc_before'],row['gc_after']))
    else:
        assert value['runner_sha256']==frozen['files']['runtime/native.py']['sha256']
        assert value['adapter_sha256']==manifest['adapter']['sha256']
        assert value['native_binaries']==manifest['native_binaries'] and value['versions']==manifest['native_versions']
        assert value['python_binary']==frozen['interpreter']
        assert value['native_settings']==dict(provider='CPUExecutionProvider',intra_threads=1,inter_threads=1,sequential=True,graph_optimizations='all',spinning=False)
        assert value['flags']=={k:'1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS']}
        assert value['numeric_libraries'] and any('onnxruntime' in p for p in value['numeric_libraries'])
        for name,wanted in value['numeric_libraries'].items():assert frozen['external'][name]==wanted,name


def resource_records(run,samples):
    assert run['complete'] is True and run['code']==0 and 'error' not in run
    assert type(run['samples']) is int and run['samples']==len(samples) and samples
    assert 0<run['seconds']<LIMITS['seconds'] and run['ended']>run['started']
    assert run['preflight_available']>=LIMITS['preflight'] and run['preflight_disk']>=LIMITS['preflight_disk']
    assert run['members'][str(run['child']['pid'])]==run['child']['birth']
    assert run['accounting']['valid'] is True and math.isfinite(run['accounting']['foreign_cpu_fraction']) and run['accounting']['foreign_cpu_fraction']>=0
    previous=0.;peak=0;seen={};gaps=[]
    for row in samples:
        check_sample(row);assert previous<row['seconds']<run['seconds'];gaps.append(row['seconds']-previous);previous=row['seconds']
        assert len({m['pid'] for m in row['members']})==len(row['members'])
        for member in row['members']:
            assert member['birth']==run['members'][str(member['pid'])]>=run['child']['birth']
            seen[str(member['pid'])]=member['birth']
        peak=max(peak,sum(m['rss'] for m in row['members']))
    gaps.append(run['seconds']-previous)
    assert max(gaps)<10 and min(gaps)>=0 and peak==run['peak_rss'] and seen==run['members']
    return dict(samples=len(samples),peak_rss=peak,min_available=min(r['available'] for r in samples),min_disk=min(r['disk'] for r in samples),max_gap=max(gaps),accounting=run['accounting'])


def refuses(action):
    try:action()
    except (AssertionError,KeyError,TypeError,ValueError,IndexError):return
    raise AssertionError('Damaged evidence accepted')


def main():
    base=BASE/'collected';receipt=read(base/'collection.json');assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
    frozen=read(base/'frozen.json')
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in read(base/'preparation.json')['bindings'].items():assert pin(ROOT/name)==wanted,name
    state=read(base/'campaign/identity.json');assert state['complete'] is True and state['code']==0 and 'error' not in state
    assert state['frozen']==pin(base/'frozen.json') and state['limits']==LIMITS and 0<state['seconds']<LIMITS['campaign_seconds']
    wanted=[(phase,family,engine) for phase in ['conformance','timing'] for family,engine in schedule(phase)]
    assert [(r['phase'],r['family'],r['engine']) for r in state['runs']]==wanted
    gate=read(base/'campaign/conformance-gate.json');assert gate['passed'] is True and gate['frozen']==state['frozen']
    assert gate['identity']==pin(base/'campaign/conformance-identity.json')
    assert read(base/'campaign/conformance-identity.json')['runs']==state['runs'][:6]
    observations=[];refusals=0;previous=state['started']
    for run in state['runs']:
        assert previous<=run['started']<run['ended'];previous=run['ended']
        folder=base/run['output'];manifest=read(base/'manifests'/(run['family']+'.json'));value=read(folder/'worker/result.json')
        validate_records(value,manifest,run['phase']);worker_identity(value,manifest,frozen,base,run['engine'])
        samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        resources=resource_records(run,samples)
        for i,row in enumerate(value['records']):assert read(folder/'worker'/f'{i:03}.json')==row
        if run['phase']=='conformance':
            assert gate['workers'][run['name']]==pin(folder/'worker/result.json')
            for change in [lambda v:v.update(runner_sha256='0'*64),lambda v:v.update(manifest_sha256='0'*64),lambda v:v.update(engine='invalid')]:
                damaged=copy.deepcopy(value);change(damaged);refuses(lambda:worker_identity(damaged,manifest,frozen,base,run['engine']));refusals+=1
            for change in [lambda s:s.update(samples=s['samples']+1),lambda s:s.update(peak_rss=-1),lambda s:s.update(code=1)]:
                damaged=copy.deepcopy(run);change(damaged);refuses(lambda:resource_records(damaged,samples));refusals+=1
            if run['family']=='whisper' and run['engine']=='ort':
                for case,row in zip(manifest['cases'],value['records'],strict=True):
                    actual=np.load(folder/'worker'/(case['name']+'.features.npy'),allow_pickle=False)
                    expected=np.load(base/'assets'/case['features']['path'],allow_pickle=False)
                    assert actual.dtype==expected.dtype==np.float32 and actual.shape==expected.shape==(1,128,3000) and np.isfinite(actual).all()
                    difference=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
                    assert row['frontend']==dict(values=384000,max_abs=float(difference.max()),failed=int((difference>1e-5).sum()),bits_equal=actual.tobytes()==expected.tobytes(),sha256=hashlib.sha256(actual.tobytes()).hexdigest())
        observations.append(dict(name=run['name'],phase=run['phase'],family=run['family'],engine=run['engine'],resource=resources,value=value))
    assert previous<=state['ended']
    table=[]
    for family in FAMILIES:
        manifest=read(base/'manifests'/(family+'.json'));groups=[(c['name'],[c]) for c in manifest['cases']] if family=='pyannote' else [('complete-corpus',manifest['cases'])]
        for name,cases in groups:
            row=dict(family=family,name=name,audio_seconds=sum(c['samples'] for c in cases)/16000)
            names={c['name'] for c in cases}
            for engine in ['managed','ort']:
                visits=[]
                for observation in observations:
                    if (observation['phase'],observation['family'],observation['engine'])==('timing',family,engine):
                        totals=[math.fsum(r['seconds'] for r in observation['value']['records'] if r['pass']==p and r['name'] in names) for p in [1,2,3]]
                        visits.append(dict(process=observation['name'],passes=totals,mean=statistics.mean(totals)))
                assert len(visits)==2;row[engine]=dict(visits=visits,seconds=statistics.mean(v['mean'] for v in visits))
                row[engine]['rtf']=row[engine]['seconds']/row['audio_seconds']
            row['ratio']=row['managed']['seconds']/row['ort']['seconds'];table.append(row)
    counts={phase:sum(len(o['value']['records']) for o in observations if o['phase']==phase) for phase in ['conformance','timing']}
    assert counts==dict(conformance=88,timing=704)
    write(BASE/'audit.json',dict(passed=True,table=table,observations=observations,refusals=refusals,counts=counts,births=receipt['births'],frozen=pin(base/'frozen.json')))
    print(json.dumps(dict(passed=True,table=table,refusals=refusals,counts=counts)))


if __name__=='__main__':main()
