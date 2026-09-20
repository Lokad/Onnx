"""Audit all actual AMD outputs, both complete references, tests and process records."""
from pathlib import Path
import sys, re, subprocess, xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tests/pyannote/filterbank-precision'))
from shared import *
from routes import scalar_window, direct_tables, direct_fourier
from vm import ssh


def resources(base,state,names,cpu,successful=True):
    assert state['complete'] and state['code']==(0 if successful else 1)
    if successful:assert not state.get('error')
    assert [r['name'] for r in state['runs']]==names
    births=[state['supervisor']];rows=[];previous_end=0
    for run in state['runs']:
        assert run['complete'] and (run['code']==0 if successful else run['code']!=0)
        assert previous_end<=run['started']<=run['ended'];previous_end=run['ended']
        assert 0<run['seconds']<600 and run['preflight_available']>=4*1024**3 and run['preflight_disk']>=2*1024**3
        births.extend(dict(pid=int(pid),birth=birth) for pid,birth in run['members'].items())
        assert run['members'][str(run['child']['pid'])]==run['child']['birth']
        samples=[json.loads(line) for line in (base/(run['name']+'.samples.jsonl')).read_text().splitlines()]
        assert samples and len(samples)==run['samples'];previous=0
        for sample in samples:
            assert previous<=sample['seconds']<=run['seconds'] and sample['seconds']-previous<10;previous=sample['seconds']
            assert sample['available']>=1024**3 and sum(m['rss'] for m in sample['members'])<2*1024**3
            if cpu==2:assert sample['disk']>=1024**3
            for m in sample['members']:assert m['affinity']==[cpu] and run['members'][str(m['pid'])]==m['birth']
        assert run['seconds']-previous<10
        rows.append(dict(name=run['name'],seconds=run['seconds'],samples=len(samples),peak_rss=max(sum(m['rss'] for m in s['members']) for s in samples)))
    return births,rows


def test_results(path,passed):
    ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'};root=ET.parse(path).getroot()
    counters=root.find('t:ResultSummary/t:Counters',ns).attrib;cases=root.findall('t:Results/t:UnitTestResult',ns)
    assert counters['total']==counters['executed']=='99' and counters['passed']==str(passed) and counters['failed']==str(99-passed)
    assert len(cases)==99 and sum(c.attrib['outcome']=='Passed' for c in cases)==passed
    failed=[c.attrib['testName'] for c in cases if c.attrib['outcome']!='Passed']
    assert all('Community1CliTests' in name for name in failed)
    if passed==99:assert sum('FrameMeanRemoval' in c.attrib['testName'] for c in cases)==2
    return dict(counters=counters,failed=failed)


def main():
    own=psutil_module().Process();own.cpu_affinity([0])
    base=ROOT/'artifacts/wespeaker-frame-product-amd-v4-20260920';collected=base/'collected';result=collected/'result'
    old=ROOT/'artifacts/wespeaker-frame-product-amd-v3-20260920/collected'
    reference=ROOT/'artifacts/wespeaker-frame-amd-reference-20260920';spec=read(reference/'manifest.json');proof=ROOT/'artifacts/wespeaker-precision-20260920'
    verify(spec['files']);assert spec['interpreter']==pin(sys.executable)
    for path,wanted in spec['numeric'].items():assert pin(path)==wanted
    remote_births=[];all_resources=[]
    for folder,names,successful in [(old,['tests'],False),(collected,['cli-build','tests','consumer-build','dc','corpus'],True)]:
        receipt=read(folder/'collection.json');assert receipt['terminal']
        for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
        payload=read(folder/'payload.json')
        for name,wanted in payload['files'].items():assert pin(folder/name)==wanted,name
        state=read(folder/'result/run.json');assert state['source']=='1d10d22f73282bb00630371887ca3719d2e1553b' and state['sdk']=='10.0.204'
        births,rows=resources(folder/'result',state,names,2,successful);assert births==receipt['births']
        remote_births+=births;all_resources.append(dict(attempt=folder.parent.name,stages=rows))
    source=collected/'source';payload=read(collected/'payload.json')
    count=0
    for name in payload['files']:
        if not name.startswith('source/'):continue
        relative=name[len('source/'):];expected=subprocess.check_output(['git','rev-parse','1d10d22:'+relative],cwd=ROOT,text=True).strip()
        raw=(collected/name).read_bytes();assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==expected;count+=1
    assert count==343
    remote_state=read(result/'run.json')
    for phase in ['dc','corpus']:
        record=read(result/phase/'result.json');runtime=record['runtime'];run=next(r for r in remote_state['runs'] if r['name']==phase)
        assert record['complete'] and runtime==dict(pid=run['child']['pid'],framework='10.0.8',affinity=4,processor_count=1)
        for item in record['loaded']:
            local=collected/Path(item['file']).relative_to('/dev/shm/onnx-frame-product-v4-20260920')
            assert pin(local)==item['pin']==pin(result/'product-bin'/local.name)
    dc=read(result/'dc/result.json');assert all(r['failed']==0 and r['maximum']==0 for r in dc['rows']) and [r['length'] for r in dc['rows']]==[560,16000]
    successful_tests=test_results(result/'trx/affected.trx',99);failed_tests=test_results(old/'result/trx/affected.trx',92)
    assert (old/'result/tests.stdout').read_text().count('Repository root with Lokad.Onnx.slnx not found.')==7
    local_state=read(reference/'run.json');local_births,local_resources=resources(reference,local_state,['numpy','torch'],0)
    assert local_state['manifest']==pin(reference/'manifest.json') and local_state['limits']==LIMITS and all(absent(b) for b in local_births)
    refs={engine:read(reference/engine/'result.json') for engine in ['numpy','torch']}
    for engine,record in refs.items():
        runtime=record['runtime'];run=next(r for r in local_state['runs'] if r['name']==engine)
        assert record['complete'] and record['manifest']==pin(reference/'manifest.json') and [r['name'] for r in record['records']]==[c['name'] for c in spec['cases']]
        assert runtime['pid']==run['child']['pid'] and runtime['birth']==run['child']['birth'] and runtime['affinity']==[0]
        assert runtime['numpy']=='2.2.4' and runtime['torch']==('2.11.0+cpu' if engine=='torch' else None) and runtime['blas_threads']==1 and not runtime['native_ort_loaded']
        if engine=='torch':assert 'BLAS_INFO=mkl' in runtime['torch_config'] and re.search(r'mkl_get_max_threads\(\)\s*:\s*1\b',runtime['torch_parallel'])
        for path,wanted in runtime['libraries'].items():assert spec['numeric'][path]==wanted==pin(path)
        assert not (reference/(engine+'.stderr')).read_text().strip()
    corpus=read(result/'corpus/result.json');assert [r['name'] for r in corpus['rows']]==[c['name'] for c in spec['cases']]
    table_metrics={}
    for name,stem,shape in [('Window','window',(400,)),('MelWeights','mel',(80,256))]:
        actual=np.fromfile(result/'corpus'/(name+'.f32'),dtype='<f4').reshape(shape)
        assert pin(result/'corpus'/(name+'.f32'))==corpus['tables'][name]==dc['tables'][name]
        assert np.array_equal(actual,np.load(ROOT/spec[stem]))
        before=np.fromfile(proof/'inputs'/(stem+'.f32'),dtype='<f4').reshape(shape)
        table_metrics[name]=dict(different=int((actual!=before).sum()),max_abs=float(np.max(np.abs(actual.astype(float)-before))),amd=corpus['tables'][name],windows=pin(proof/'inputs'/(stem+'.f32')))
    window=np.load(ROOT/spec['window']);tables=direct_tables();stages=[];scalars=[];checks=[];duplicate=[];seen={};numeric_bytes=0
    for index,case in enumerate(spec['cases']):
        expected=case['shapes'];pcm=np.load(ROOT/case['input']);assert pcm.dtype==np.float32 and pcm.size==case['samples']
        actual_path=result/'corpus'/(case['name']+'.f32');entry=corpus['rows'][index]
        assert entry['shape']==expected['features'] and entry['output']==pin(actual_path)
        assert entry['input']==hashlib.sha256(pcm.tobytes()).hexdigest()==pin(collected/'reference/inputs'/(case['name']+'.f32'))['sha256']
        actual=np.fromfile(actual_path,dtype='<f4').reshape(expected['features']);assert np.isfinite(actual).all()
        arrays={};case_pins={}
        for engine in ['numpy','torch']:
            record=refs[engine]['records'][index];assert record['input_unchanged'] and record['coefficients_unchanged'] and list(record['stages'])==list(STAGES)
            arrays[engine]={}
            for stage in STAGES:
                path=reference/engine/case['name']/(stage+'.npy');meta=record['stages'][stage]
                assert pin(path)==meta['pin'] and meta['shape']==expected[stage]
                value=np.load(path,allow_pickle=False);check_array(value,expected[stage]);arrays[engine][stage]=value
                case_pins[engine+'/'+stage]=pin(path);numeric_bytes+=value.nbytes
        for stage in STAGES:stages.append(dict(name=case['name'],stage=stage,**metric(arrays['numpy'][stage],arrays['torch'][stage],1e-8)))
        for frame in sorted({0,expected['features'][1]//2,expected['features'][1]-1}):
            value=scalar_window(pcm,window,frame);real,imaginary=direct_fourier(value,tables)
            for engine in ['numpy','torch']:
                for stage,target in [('windowed',value),('real',real),('imaginary',imaginary)]:
                    scalars.append(dict(name=case['name'],engine=engine,frame=frame,stage=stage,**metric(arrays[engine][stage][frame],target,1e-8)))
        for engine in ['numpy','torch']:
            for policy,target in [('amd',arrays[engine]['features']),('native',np.load(ROOT/case['old_reference']/engine/case['old_name']/'features.npy'))]:
                checks.append(dict(name=case['name'],target=policy+'-'+engine,**metric(actual,target,1e-4)))
        native=load_baseline(case['baselines']['native']);windows=np.fromfile(ROOT/'artifacts/wespeaker-frame-product-local-20260920/corpus'/(case['name']+'.f32'),dtype='<f4').reshape(expected['features'])
        checks.append(dict(name=case['name'],target='native-float',**metric(actual,native,1e-4)))
        checks.append(dict(name=case['name'],target='windows-product',bits_equal=bool(np.array_equal(actual.view(np.uint32),windows.view(np.uint32))),**metric(actual,windows,1e-4)))
        case_pins['actual']=pin(actual_path)
        if entry['input'] in seen:
            before,pins=seen[entry['input']];assert pins==case_pins;duplicate.append(dict(name=case['name'],original=before,arrays=len(pins)))
        else:seen[entry['input']]=(case['name'],case_pins)
    assert all(row['failed']==0 for row in stages+scalars)
    summary={}
    for target in ['amd-numpy','amd-torch','native-numpy','native-torch','native-float','windows-product']:
        rows=[r for r in checks if r['target']==target]
        summary[target]=dict(arrays=len(rows),values=sum(r['values'] for r in rows),failed=sum(r['failed'] for r in rows),max_scaled=max(r['max_scaled'] for r in rows))
    assert all(summary[k]['failed']==0 for k in ['amd-numpy','amd-torch','native-numpy','native-torch'])
    assert len(stages)==371 and len(duplicate)==3 and all(r['values']==3266560 for r in summary.values())
    status=json.loads(ssh('''import sys,json
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
births=%r
for b in births:
 try:assert psutil.Process(b['pid']).create_time()!=b['birth']
 except psutil.NoSuchProcess:pass
print(json.dumps(dict(terminal=True,count=len(births))))
'''%remote_births));assert status['terminal']
    warnings={r['name']:sorted(set(line.strip() for line in (result/(r['name']+'.stdout')).read_text().splitlines() if 'warning' in line.lower())) for r in remote_state['runs']}
    observation=dict(passed=True,source=remote_state['source'],source_blobs=count,tests=successful_tests,failed_attempt=failed_tests,dc=dc['rows'],tables=table_metrics,
        arrays=742,numeric_bytes=numeric_bytes,stages=stages,scalars=scalars,checks=checks,summary=summary,duplicates=duplicate,
        windows_bit_equal_arrays=sum(r.get('bits_equal',False) for r in checks),remote_births=remote_births,local_births=local_births,
        auditor=dict(pid=own.pid,birth=own.create_time()),resources=all_resources,reference_resources=local_resources,warnings=warnings,
        assemblies={p.name:pin(p) for p in (result/'product-bin').iterdir() if p.name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','Lokad.Onnx.Backend.Tests.dll']},
        bindings={str(p.relative_to(ROOT)):pin(p) for p in [collected/'collection.json',old/'collection.json',reference/'manifest.json']})
    write(base/'audit.json',observation);print(json.dumps(dict(passed=True,summary=summary,max_reference=max(r['max_scaled'] for r in stages),max_scalar=max(r['max_scaled'] for r in scalars),windows_bit_equal_arrays=observation['windows_bit_equal_arrays'])))


if __name__=='__main__':main()
