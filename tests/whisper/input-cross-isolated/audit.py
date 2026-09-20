"""Independently validate every isolated worker and the complete four-way corpus."""
from pathlib import Path
import argparse,json,math,sys
import numpy as np
from protocol import schedule,coverage
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'input-cross'))
import audit as original
from common import ROOT,CORE,pin,read,write,verify

def header(result,job,manifest_sha):
    assert result['complete'] is True and result['engine']==job['engine'] and result['request_index']==job['request']
    assert result['manifest_sha256']==manifest_sha and result['flags']=={} and len(result['records'])==2
    assert [r['request'] for r in result['records']]==[job['request']]*2
    assert [r['name'] for r in result['records']]==[job['name']]*2
    assert [r['kind'] for r in result['records']]==list(original.KINDS[job['engine']])
    assert result['records'][0]['baseline_matches'] is True and result['records'][1]['baseline_matches'] is None

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.artifact.resolve();manifest=base/'manifest.json';spec=read(manifest);verify(spec)
    assert spec['protocol']=='whisper-input-cross-isolated-case-v3' and spec['workers']==42 and spec['cases']==21 and spec['new_encoder_arrays']==84
    assert spec['scaled_error_limit']==1e-4
    state=read(base/'campaign.json');assert state['complete'] is True and state['code']==0 and not state.get('error')
    assert state['manifest_sha256']==pin(manifest)['sha256'] and original.absent(state['supervisor'])
    jobs=[run['job'] for run in state['runs']];coverage(jobs,spec)
    assert all(a['ended']<=b['started'] for a,b in zip(state['runs'],state['runs'][1:]))
    preflight=[json.loads(line) for line in (base/'preflight.jsonl').read_text().splitlines()]
    assert preflight and all(a['time']<=b['time'] for a,b in zip(preflight,preflight[1:]))
    arrays={};resources=[];files={};first={};observed_jobs=[]
    for run,job in zip(state['runs'],jobs):
        observations=[r for r in preflight if r['job']==job['id']];assert observations
        assert observations[-1]['available']==run['preflight_available'] and observations[-1]['time']<=run['started']
        assert all(v['available']<spec['limits']['preflight_available'] for v in observations[:-1])
        assert observations[-1]['time']-observations[0]['time']<=spec['preflight_wait_seconds']+6
        observed_jobs.extend([job['id']]*len(observations))
        folder=base/'process'/job['id'];samples=[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]
        assert run['supervisor']==state['supervisor']
        resource=original.resources(run,samples,spec);assert all(original.absent(i) for i in resource['births'])
        resources.append(dict(job=job,**resource))
        output=base/'outputs'/job['id'];result=read(output/'result.json');header(result,job,pin(manifest)['sha256'])
        engine=job['engine'];index=job['request'];item=spec['requests'][index];assert item['request']==index and item['name']==job['name']
        if engine=='managed':
            assert result['context_lifecycle']=='fresh-per-call-isolated-case'
            assert result['core_sha256']==CORE and result['probe_sha256']==pin(base/'bin/WhisperInputCross.dll')['sha256']
            assert result['runtime']=='10.0.12' and result['affinity']==4 and result['processor_count']==1
            assert result['native_loaded'] is False and result['packed_weight_bytes']==256*1024**2
        else:
            assert result['native_runtime']==spec['native_runtime'] and result['affinity']==[2]
            assert result['threads']==1 and result['sequential'] is True and result['all_optimizations'] is True and result['spinning'] is False
            assert result['modules'] and all(spec['native_runtime']['files'].get(k)==v for k,v in result['modules'].items())
        expected=original.source(item['native_hidden']);expected_files={'result.json'};held={}
        for row,kind in zip(result['records'],original.KINDS[engine]):
            name=f"{index:02}-{item['name']}-{kind}.f32";assert row['file']==name;path=output/name
            identity=pin(path);assert identity==dict(bytes=1500*1280*4,sha256=row['sha256'])
            value=np.fromfile(path,dtype='<f4').reshape(1,1500,1280)
            feature='managed_features' if kind in ['MM','NM'] else 'native_features';original.source(item[feature])
            original.validate_record(row,index,item,kind,value,expected,item[feature]['raw_sha256'])
            held[kind]=original.raw(value)
            if index==0:first[kind]=held[kind]
            if index==20:assert held[kind]==first[kind],'Cross-process first-case repeat changed'
            arrays[(index,kind)]=path;expected_files.add(name)
        assert result['held_outputs']==held and {p.name for p in output.iterdir()}==expected_files
    assert [r['job'] for r in preflight]==observed_jobs
    assert {p.name for p in (base/'outputs').iterdir()}=={j['id'] for j in jobs}
    assert {p.name for p in (base/'process').iterdir()}=={j['id'] for j in jobs}
    rows=[]
    for index,item in enumerate(spec['requests']):
        values=[np.fromfile(arrays[(index,k)],dtype='<f4').reshape(1,1500,1280) for k in ['MM','NN','MN','NM']]
        rows.append(dict(request=index,name=item['name'],**original.decompose(*values)))
    aggregate={}
    for term in original.TERMS:
        cells=[r['terms'][term] for r in rows]
        aggregate[term]=dict(arrays=21,failed_arrays=sum(c['failed_values']>0 for c in cells),values=sum(c['values'] for c in cells),
            failed_values=sum(c['failed_values'] for c in cells),max_abs=max(c['max_abs'] for c in cells),max_scaled=max(c['max_scaled'] for c in cells),
            l2=math.sqrt(sum(c['sum_squares'] for c in cells)))
    verify(spec)
    for directory in ['outputs','process']:
        for path in (base/directory).rglob('*'):
            if path.is_file():files[path.relative_to(base).as_posix()]=pin(path)
    for name in ['campaign.json','preflight.jsonl']:files[name]=pin(base/name)
    write(args.output,dict(schema=1,structural_passed=True,manifest=pin(manifest),workers=42,arrays=84,baseline_bridges=42,
                          resources=resources,aggregate=aggregate,rows=rows,files=files,
                          scope='Full-corpus input/engine localization in finite workers; held outputs within each pair, repeat across processes'))
    print(json.dumps(aggregate,indent=2))

if __name__=='__main__':main()
