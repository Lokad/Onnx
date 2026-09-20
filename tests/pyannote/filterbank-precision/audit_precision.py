"""Independent complete-array audit of the frozen fixed-coefficient experiment."""
import argparse, re
from shared import *
from generate import generate
from routes import scalar_window,direct_tables,direct_fourier


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);a=p.parse_args();base=Path(a.artifact).resolve()
    ps=psutil_module();own=ps.Process();own.cpu_affinity([0])
    spec=read(base/'manifest.json');verify(spec['files']);state=read(base/'run.json')
    assert spec['limits']==LIMITS and spec['variants']==list(VARIANTS) and spec['stages']==list(STAGES)
    assert spec['reference_limit']==REFERENCE_LIMIT==1e-8 and spec['original_limit']==ORIGINAL_LIMIT==1e-4 and spec['rounded_limit']==1e-6
    assert spec['interpreter']==pin(sys.executable) and spec['tests']==pin(base/'tests.json') and read(base/'tests.json')['code']==0
    for path,wanted in spec['numeric'].items():assert pin(path)==wanted,path
    assert state['manifest']==pin(base/'manifest.json')
    samples={r['name']:[json.loads(line) for line in (base/(r['name']+'.samples.jsonl')).read_text().splitlines()] for r in state['runs']}
    births,resources=resource_checks(state,samples);assert all(absent(b) for b in births)
    for variant in VARIANTS:assert (base/'app'/(variant+'.cs')).read_text()==generate(PRODUCT.read_bytes(),variant)
    log=(base/'build.stdout').read_text();assert 'Build succeeded.' in log and re.search(r'0 Error\(s\)',log)
    warnings=sorted(set(re.findall(r'^.*warning [A-Z]+\d+.*$',log,re.MULTILINE)))
    data={engine:read(base/engine/'result.json') for engine in ['managed','numpy','torch']}
    cases=spec['cases'];assert len(cases)==53 and len({c['name'] for c in cases})==53
    for engine,result in data.items():
        runtime=result['runtime'];run=next(r for r in state['runs'] if r['name']==engine)
        assert result['complete'] and result['manifest']==state['manifest'] and runtime['pid']==run['child']['pid']
        assert not runtime['native_ort_loaded'] and [r['name'] for r in result['records']]==[c['name'] for c in cases]
        assert (base/(engine+'.stdout')).read_text().splitlines()==[c['name'] for c in cases]
        if engine=='managed':
            assert runtime['framework']=='10.0.12' and runtime['affinity']==runtime['processor_count']==1
            assert result['core']==pin(CORE) and result['core']['sha256']==CORE_SHA
            assert result['producer']==pin(base/'app/bin/Release/net10.0/Precision.dll')
        else:
            assert runtime['birth']==run['child']['birth'] and runtime['affinity']==[0] and runtime['numpy']=='2.2.4' and runtime['blas_threads']==1
            assert runtime['torch']==('2.11.0+cpu' if engine=='torch' else None)
            if engine=='torch':assert 'BLAS_INFO=mkl' in runtime['torch_config'] and re.search(r'mkl_get_max_threads\(\)\s*:\s*1\b',runtime['torch_parallel'])
            for path,wanted in runtime['libraries'].items():assert spec['numeric'][path]==wanted and pin(path)==wanted
    for run in state['runs']:assert not (base/(run['name']+'.stderr')).read_text().strip()
    window=np.load(ROOT/spec['window']);mel=np.load(ROOT/spec['mel']);tables=direct_tables()
    reference_checks=[];all_checks=[];scalars=[];controls=[];comparisons=[];stage_errors=[];invariants=[];duplicates=[]
    identities={};numeric_bytes=0;array_count=0
    for index,case in enumerate(cases):
        pcm=np.load(ROOT/case['input'],allow_pickle=False);expected=shapes(int(pcm.size));assert case['shapes']==expected and pcm.size==case['samples']
        raw=base/'inputs'/(case['name']+'.f32');assert raw.read_bytes()==pcm.tobytes();input_hash=pin(raw)['sha256']
        arrays={};case_pins={}
        for engine in ['numpy','torch']:
            record=data[engine]['records'][index];assert record['input_unchanged'] and record['coefficients_unchanged'] and list(record['stages'])==list(STAGES)
            arrays[engine]={}
            for stage,row in record['stages'].items():
                path=base/engine/case['name']/(stage+'.npy');assert pin(path)==row['pin'] and row['shape']==expected[stage]
                value=np.load(path,allow_pickle=False);check_array(value,expected[stage]);arrays[engine][stage]=value
                case_pins[engine+'/'+stage]=pin(path);numeric_bytes+=value.nbytes;array_count+=1
        record=data['managed']['records'][index];assert record['pcm']==input_hash and [r['variant'] for r in record['rows']]==list(VARIANTS)
        for row in record['rows']:
            variant=row['variant'];assert row['input_unchanged'] and row['coefficients_unchanged'] and list(row['stages'])==list(STAGES)
            directory=base/'managed'/case['name']/variant;arrays[variant]={}
            assert {p.name for p in directory.iterdir()}=={s+'.f64' for s in STAGES}|{'features.f32'}
            for stage,entry in row['stages'].items():
                path=directory/(stage+'.f64');assert pin(path)==entry['pin'] and entry['shape']==expected[stage]
                value=np.fromfile(path,dtype='<f8').reshape(expected[stage]);check_array(value,expected[stage]);arrays[variant][stage]=value
                case_pins[variant+'/'+stage]=pin(path);numeric_bytes+=value.nbytes;array_count+=1
            path=directory/'features.f32';assert pin(path)==row['features'];value=np.fromfile(path,dtype='<f4').reshape(expected['features'])
            np.testing.assert_array_equal(value.astype(np.float64),arrays[variant]['features']);numeric_bytes+=value.nbytes;array_count+=1
            case_pins[variant+'/float']=pin(path)
        for stage in STAGES:
            reference_checks.append(dict(name=case['name'],stage=stage,**comparison(arrays['numpy'][stage],arrays['torch'][stage],REFERENCE_LIMIT)))
            for ref in ['numpy','torch']:
                target=arrays[ref][stage].astype(np.float32).astype(np.float64) if stage=='features' else arrays[ref][stage]
                all_checks.append(dict(name=case['name'],reference=ref,stage=stage,**comparison(arrays['All'][stage],target,1e-6 if stage=='features' else REFERENCE_LIMIT)))
                for variant in VARIANTS:
                    stage_errors.append(dict(name=case['name'],variant=variant,reference=ref,stage=stage,**comparison(arrays[variant][stage],arrays[ref][stage])))
        for frame in sorted({0,expected['windowed'][0]//2,expected['windowed'][0]-1}):
            value=scalar_window(pcm,window,frame);real,imaginary=direct_fourier(value,tables)
            for engine in ['numpy','torch','All']:
                for stage,target in [('windowed',value),('real',real),('imaginary',imaginary)]:
                    scalars.append(dict(name=case['name'],engine=engine,frame=frame,stage=stage,**comparison(arrays[engine][stage][frame],target,REFERENCE_LIMIT)))
        original=load_baseline(case['baselines']['windows-default']);native=load_baseline(case['baselines']['native'])
        actual=arrays['Original']['features'].astype(np.float32)
        controls.append(dict(name=case['name'],bits_equal=bool(np.array_equal(actual.view(np.uint32),original.view(np.uint32))),**comparison(actual,original)))
        for variant in VARIANTS:
            feature=arrays[variant]['features']
            targets={ref:arrays[ref]['features'] for ref in ['numpy','torch']}
            targets['native']=native
            for ref in ['numpy','torch']:targets['old-'+ref]=np.load(ROOT/case['old_reference']/ref/case['old_name']/'features.npy',allow_pickle=False)
            for ref,target in targets.items():comparisons.append(dict(name=case['name'],corpus=case['corpus'],variant=variant,reference=ref,**comparison(feature,target)))
            zero=case['corpus']=='unit' and (case['name'].startswith(('unit-silence-','unit-dc-')) or expected['features'][1]==1)
            centered=float(np.abs(feature.mean(axis=1)).max());maximum=float(np.abs(feature).max())
            invariants.append(dict(name=case['name'],variant=variant,mean=centered,zero_expected=zero,maximum=maximum,passed=centered<=1e-5 and (not zero or maximum<=1e-9)))
        # Native error against both fixed-managed-coefficient references is retained too.
        for ref in ['numpy','torch']:comparisons.append(dict(name=case['name'],corpus=case['corpus'],variant='Native',reference=ref,**comparison(native,arrays[ref]['features'])))
        if input_hash in identities:
            original_name,pins=identities[input_hash];equal=case_pins==pins
            duplicates.append(dict(name=case['name'],original=original_name,arrays=len(case_pins),equal=equal))
        else:identities[input_hash]=(case['name'],case_pins)
        print(case['name'],flush=True)
    summary=[]
    for variant in [*VARIANTS,'Native']:
        for corpus in ['unit','five','dialogue','all']:
            for ref in ['numpy','torch','native','old-numpy','old-torch']:
                rows=[r for r in comparisons if r['variant']==variant and r['reference']==ref and (corpus=='all' or r['corpus']==corpus)]
                if rows:summary.append(dict(variant=variant,corpus=corpus,reference=ref,**aggregate(rows)))
    qualified=[]
    valid=all(r['bits_equal'] for r in controls) and all(r['failed']==0 for r in reference_checks+all_checks+scalars) and all(r['equal'] for r in duplicates)
    for variant in VARIANTS[1:]:
        passed=valid and all(r['passed'] for r in invariants if r['variant']==variant)
        for ref in ['numpy','torch']:
            row=next(r for r in summary if (r['variant'],r['corpus'],r['reference'])==(variant,'all',ref))
            old=next(r for r in summary if (r['variant'],r['corpus'],r['reference'])==('Original','all',ref))
            passed &= row['failed']==0 and row['squared_error']<old['squared_error']
        if passed:qualified.append(variant)
    nominated=min(qualified,key=lambda v:(3 if v=='All' else 1,sum(r['squared_error'] for r in summary if r['variant']==v and r['corpus']=='all' and r['reference'] in ['numpy','torch']))) if qualified else None
    result=dict(structural_passed=True,diagnostic_passed=valid,manifest=pin(base/'manifest.json'),source=spec['source'],arrays=array_count,numeric_bytes=numeric_bytes,
                resources=resources,births=births,auditor=dict(pid=own.pid,birth=own.create_time()),warnings=warnings,
                controls=controls,references=reference_checks,all_double=all_checks,scalars=scalars,stage_errors=stage_errors,
                comparisons=comparisons,summary=summary,invariants=invariants,duplicates=duplicates,qualified=qualified,nominated=nominated,
                files={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()})
    write(base/'audit.json',result)
    print(json.dumps(dict(diagnostic_passed=valid,nominated=nominated,qualified=qualified,arrays=array_count,numeric_bytes=numeric_bytes,
                         control_failures=sum(not r['bits_equal'] for r in controls),reference_failures=sum(r['failed'] for r in reference_checks),all_failures=sum(r['failed'] for r in all_checks))))


if __name__=='__main__':main()
