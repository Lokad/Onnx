"""Independently validate the four encoder cells and their exact decomposition."""
from pathlib import Path
import argparse,hashlib,math,sys
import numpy as np
from common import ROOT,CORE,pin,read,write,verify
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil

KINDS={'managed':('MM','MN'),'native':('NN','NM')}
TERMS=('original_MM-NN','engine_MM-NM','engine_MN-NN','input_MM-MN','input_NM-NN','interaction')

def raw(v):return hashlib.sha256(v.tobytes()).hexdigest()

def source(record):
    path=ROOT/record['file'];value=np.load(path,allow_pickle=False) if record['format']=='npy' else np.fromfile(path,dtype='<f4').reshape(record['shape'])
    assert value.dtype==np.float32 and list(value.shape)==record['shape'] and np.isfinite(value).all()
    assert raw(value)==record['raw_sha256'];return value

def metrics(delta,denominator):
    delta=np.asarray(delta,dtype=np.float64);denominator=np.asarray(denominator,dtype=np.float64)
    assert delta.shape==denominator.shape and np.isfinite(delta).all() and np.isfinite(denominator).all() and (denominator>=1).all()
    scaled=np.abs(delta)/denominator
    return dict(values=int(delta.size),failed_values=int(np.count_nonzero(scaled>1e-4)),max_abs=float(np.abs(delta).max()),
                max_scaled=float(scaled.max()),sum_squares=float(np.sum(delta*delta)),l2=float(np.linalg.norm(delta.ravel())))

def decompose(mm,nn,mn,nm):
    arrays=[np.asarray(v,dtype=np.float64) for v in [mm,nn,mn,nm]]
    assert all(v.shape==arrays[0].shape and np.isfinite(v).all() for v in arrays)
    mm,nn,mn,nm=arrays;denominator=np.maximum(1,np.abs(nn))
    deltas=[mm-nn,mm-nm,mn-nn,mm-mn,nm-nn,(mm-mn)-(nm-nn)]
    residuals=[float(np.abs(deltas[0]-(deltas[1]+deltas[4])).max()),float(np.abs(deltas[0]-(deltas[3]+deltas[2])).max())]
    assert max(residuals)<=1e-12
    return dict(terms={k:metrics(v,denominator) for k,v in zip(TERMS,deltas)},closure_max=residuals)

def validate_record(row,index,item,kind,value,expected,input_hash):
    assert row['request']==index and row['name']==item['name'] and row['kind']==kind
    assert row['shape']==[1,1500,1280] and value.shape==(1,1500,1280) and value.dtype==np.float32 and np.isfinite(value).all()
    assert row['input_sha256']==input_hash and row['inputs_unchanged'] is True and row['held_outputs_unchanged'] is True
    baseline=kind in ['MM','NN']
    assert row['baseline_matches'] is (True if baseline else None)
    if baseline:assert raw(value)==item['managed_hidden' if kind=='MM' else 'native_hidden']['raw_sha256']
    measured=metrics(value.astype(np.float64)-expected,np.maximum(1,np.abs(expected.astype(np.float64))))
    assert row['values']==measured['values'] and row['failed_values']==measured['failed_values']
    assert row['max_scaled']==measured['max_scaled'] and row['numerical_passed'] is (measured['failed_values']==0)

def absent(identity):
    try:return psutil.Process(identity['pid']).create_time()!=identity['birth']
    except psutil.NoSuchProcess:return True

def resources(state,samples,spec):
    assert state['complete'] is True and state['code']==0 and not state.get('error') and state['terminal_members'] is True
    assert state['limits']==spec['limits'] and state['preflight_available']>=spec['limits']['preflight_available']
    assert 0<state['seconds']<=spec['limits']['seconds'] and state['started']<state['ended']
    assert len(samples)==state['samples'] and samples and all(a['seconds']<b['seconds'] for a,b in zip(samples,samples[1:]))
    assert samples[0]['seconds']<5 and state['seconds']-samples[-1]['seconds']<5
    assert all(b['seconds']-a['seconds']<5 for a,b in zip(samples,samples[1:])),'Sampling gap'
    members={};peak=0
    for sample in samples:
        assert 0<=sample['seconds']<=state['seconds'] and sample['available']>=spec['limits']['available']
        assert len({p['pid'] for p in sample['members']})==len(sample['members'])
        total=0
        for process in sample['members']:
            assert process['affinity']==[2] and process['rss']>=0
            pid=str(process['pid']);birth=process['birth'];assert state['members'][pid]==birth
            assert pid not in members or members[pid]==birth;members[pid]=birth;total+=process['rss']
        peak=max(peak,total)
    assert peak==state['peak_rss']<=spec['limits']['rss'] and members==state['members']
    assert members[str(state['child']['pid'])]==state['child']['birth']
    return dict(peak_rss=peak,samples=len(samples),minimum_available=min(s['available'] for s in samples),
                births=[state['supervisor']]+[dict(pid=int(pid),birth=birth) for pid,birth in members.items()])

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();base=args.artifact.resolve();manifest=base/'manifest.json';spec=read(manifest);verify(spec)
    assert spec['cases']==len(spec['requests'])==21 and spec['new_encoder_arrays']==84 and spec['scaled_error_limit']==1e-4
    results={};telemetry={};states={};arrays={};newfiles={}
    for engine,kinds in KINDS.items():
        state=read(base/(engine+'-process')/'identity.json');states[engine]=state
        assert state['engine']==engine and state['manifest_sha256']==pin(manifest)['sha256']
        samples=[__import__('json').loads(line) for line in (base/(engine+'-process')/'samples.jsonl').read_text().splitlines()]
        telemetry[engine]=resources(state,samples,spec);assert all(absent(i) for i in telemetry[engine]['births']),'Original process still alive'
        result=read(base/engine/'result.json');results[engine]=result
        assert result['engine']==engine and result['complete'] is True and result['manifest_sha256']==pin(manifest)['sha256'] and result['flags']=={}
        assert len(result['records'])==42
        if engine=='managed':
            assert result['core_sha256']==CORE and result['probe_sha256']==pin(base/'bin/WhisperInputCross.dll')['sha256']
            assert result['runtime']=='10.0.12' and result['affinity']==4 and result['processor_count']==1
            assert result['native_loaded'] is False and result['packed_weight_bytes']==256*1024**2
        else:
            assert result['native_runtime']==spec['native_runtime'] and result['affinity']==[2]
            assert result['threads']==1 and result['sequential'] is True and result['all_optimizations'] is True and result['spinning'] is False
            assert result['modules'] and all(spec['native_runtime']['files'].get(k)==v for k,v in result['modules'].items())
        expected_files={'result.json'};first={}
        for index,item in enumerate(spec['requests']):
            assert item['request']==index
            expected=source(item['native_hidden'])
            for offset,kind in enumerate(kinds):
                row=result['records'][2*index+offset];name=f"{index:02}-{item['name']}-{kind}.f32";assert row['file']==name
                path=base/engine/name;identity=pin(path);assert identity['sha256']==row['sha256'] and identity['bytes']==1500*1280*4
                value=np.fromfile(path,dtype='<f4').reshape(1,1500,1280)
                feature='managed_features' if kind in ['MM','NM'] else 'native_features'
                source(item[feature]);validate_record(row,index,item,kind,value,expected,item[feature]['raw_sha256'])
                if index==0:first[kind]=raw(value)
                if index==20:assert raw(value)==first[kind]
                arrays[(index,kind)]=path;expected_files.add(name);newfiles[path.relative_to(base).as_posix()]=identity
        assert result['held_outputs']==first and {p.name for p in (base/engine).iterdir()}==expected_files
    assert states['managed']['ended']<=states['native']['started'],'Workers overlapped'
    rows=[]
    for index,item in enumerate(spec['requests']):
        values=[np.fromfile(arrays[(index,k)],dtype='<f4').reshape(1,1500,1280) for k in ['MM','NN','MN','NM']]
        rows.append(dict(request=index,name=item['name'],**decompose(*values)))
    aggregate={}
    for key in TERMS:
        cells=[row['terms'][key] for row in rows];squares=sum(c['sum_squares'] for c in cells)
        aggregate[key]=dict(arrays=len(cells),failed_arrays=sum(c['failed_values']>0 for c in cells),values=sum(c['values'] for c in cells),
            failed_values=sum(c['failed_values'] for c in cells),max_abs=max(c['max_abs'] for c in cells),max_scaled=max(c['max_scaled'] for c in cells),l2=math.sqrt(squares))
    verify(spec)
    for engine in KINDS:
        for path in (base/(engine+'-process')).iterdir():newfiles[path.relative_to(base).as_posix()]=pin(path)
        newfiles[engine+'/result.json']=pin(base/engine/'result.json')
    write(args.output,dict(schema=1,structural_passed=True,manifest=pin(manifest),arrays=84,baseline_bridges=42,rows=rows,aggregate=aggregate,
                          resources=telemetry,files=newfiles,scope='Input/engine localization; numerical failures remain; no timing or accuracy promotion'))
    print(__import__('json').dumps(aggregate,indent=2))

if __name__=='__main__':main()
