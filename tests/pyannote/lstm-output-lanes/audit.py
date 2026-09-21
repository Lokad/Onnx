"""Audit complete model/application comparisons without filtering timings or values."""
import collections
import json
from pathlib import Path
import statistics
import numpy as np
from prepare import ROOT, BASE, INPUT, pin, save
from run import absent


def scaled(left,right):
    assert left.shape==right.shape and np.isfinite(left).all() and np.isfinite(right).all()
    delta=np.abs(left.astype(np.float64)-right.astype(np.float64))/np.maximum(1.,np.abs(right.astype(np.float64)))
    return dict(max_scaled_error=float(delta.max(initial=0)),failed_values=int(np.count_nonzero(delta>1e-4)),values=left.size)


def main():
    assert not (BASE/'analysis.json').exists()
    spec=json.loads((BASE/'manifest.json').read_text());controls=json.loads((BASE/'run-controls.json').read_text())
    assert pin(BASE/'manifest.json')==controls['manifest'] and pin(Path(__file__).with_name('run.py'))==controls['runner']
    for name,expected in spec['files'].items():assert pin(ROOT/name)==expected,name
    state=json.loads((BASE/'processes.json').read_text())
    assert state['complete'] and state['code']==0 and absent(state['supervisor'])
    assert [r['role'] for r in state['runs']]==spec['jobs'] and len(state['runs'])==4
    manifest=json.loads(INPUT.read_text());reference=ROOT/manifest['reference']['path']
    assert pin(reference)=={k:manifest['reference'][k] for k in ('bytes','sha256')}
    native=json.loads(reference.read_text());native_cases={c['name']:c for c in native['cases']}
    crops=[c['name'] for c in manifest['cases'] if c['samples']==160000]
    outputs={};results={};comparisons=[];profiles=[];arrays=0;samples=0;identities=[state['supervisor']];native_files={str(reference.relative_to(ROOT)):pin(reference)}
    for run in state['runs']:
        index=run['index'];role=run['role'];key=f'{index}-{role}';folder=BASE/'outputs'/key
        assert run['complete'] and run['code']==0 and absent(run['worker']);identities.append(run['worker'])
        assert run['preflight']['available']>=controls['limits']['preflight_available'] and run['preflight']['disk']>=controls['limits']['disk']
        resource=[json.loads(s) for s in (BASE/'process'/key/'samples.jsonl').read_text().splitlines()]
        assert len(resource)==run['samples']>0 and max(s['rss'] for s in resource)==run['peak_rss'];samples+=len(resource)
        for s in resource:
            assert s['rss']<controls['limits']['rss'] and s['available']>=controls['limits']['available'] and s['disk']>=controls['limits']['disk']
            assert s['seconds']<controls['limits']['seconds'] and s['affinity']==[2] and s['pid']==run['worker']['pid'] and s['birth']==run['worker']['birth']
        result=json.loads((folder/'result.json').read_text());results[key]=result
        assert result['passed'] and result['inputs_and_held_outputs_unchanged'] and result['runtime']=='10.0.12'
        assert result['manifest_sha256']==pin(INPUT)['sha256'] and result['core_sha256']==spec['cores'][role]['sha256']
        assert result['data_sha256']=='e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'
        assert [(r['name'],r['model'],r['pass']) for r in result['rows']]==[(c,m,p) for c in crops for m in ('segmentation','embedding') for p in range(3)]
        groups=collections.defaultdict(list);files=set()
        for row in result['rows']:
            for kind in ('input','output'):
                info=row[kind];path=folder/info['file'];values=np.fromfile(path,dtype='<f4').reshape(info['shape'])
                assert pin(path)['sha256']==info['sha256'] and values.size==info['values'] and np.isfinite(values).all()
                files.add(info['file'])
                if kind=='output':outputs[(key,row['name'],row['model'],row['pass'])]=values;arrays+=1
            groups[(row['name'],row['model'])].append(row)
            assert row['seconds']==(row['end_ticks']-row['start_ticks'])/row['frequency']>0
            assert len(row['nodes'])==((108 if row['model']=='segmentation' else 75) if row['pass']==2 else 0)
            if row['pass']==2:
                previous=row['start_ticks'];by=collections.defaultdict(float)
                for node in row['nodes']:
                    assert previous<=node['start_ticks']<=node['end_ticks']<=row['end_ticks']
                    assert node['seconds']==(node['end_ticks']-node['start_ticks'])/row['frequency']
                    previous=node['end_ticks'];by[node['op']]+=node['seconds']
                profiles.append(dict(worker=key,name=row['name'],model=row['model'],seconds=row['seconds'],operators=dict(by)))
            wanted_name=native_cases[row['name']]['windows'][0]['scores' if row['model']=='segmentation' else 'encoded']
            path=reference.parent/wanted_name;expected=native['files'][wanted_name]
            assert pin(path)=={k:expected[k] for k in ('bytes','sha256')};native_files[str(path.relative_to(ROOT))]=pin(path)
            wanted=np.load(path,allow_pickle=False)
            assert wanted.dtype==np.float32
            check=scaled(outputs[(key,row['name'],row['model'],row['pass'])],wanted)
            comparisons.append(dict(worker=key,name=row['name'],model=row['model'],pass_index=row['pass'],reference='native',**check))
        for rows in groups.values():
            assert len({r['input']['sha256'] for r in rows})==len({r['output']['sha256'] for r in rows})==1
        assert len(files)==24 and {p.name for p in folder.iterdir()}==files|{'result.json'}
        assert [(a['name'],a['pass'],a['phase']) for a in result['applications']]==[(c['name'],p,'warmup' if p==0 else 'measured') for p in range(4) for c in manifest['cases']]
        application_first={}
        for row in result['applications']:
            actual=row['result'];wanted=next(c['expected'] for c in manifest['cases'] if c['name']==row['name'])
            assert actual['Status']==0 and wanted['status']=='Completed' and actual['Windows']==wanted['windows'] and actual['AudioDuration']==wanted['audio_seconds']
            for a,b in [('Intervals','intervals'),('ExclusiveIntervals','exclusive_intervals')]:
                assert len(actual[a])==len(wanted[b])
                for x,y in zip(actual[a],wanted[b],strict=True):
                    assert x['Speaker']==y[2] and abs(x['Start']-y[0])<=1e-12 and abs(x['End']-y[1])<=1e-12
            assert len(actual['Speakers'])==len(wanted['speakers']);maximum=0.
            for x,y in zip(actual['Speakers'],wanted['speakers'],strict=True):
                assert x['Speaker']==y['speaker'] and x['HasEmbedding']==y['has_embedding']
                a=np.array(x['Centroid']);b=np.array(y['centroid']);assert a.shape==b.shape==(256,)
                check=scaled(a,b);assert check['failed_values']==0;maximum=max(maximum,check['max_scaled_error'])
            assert maximum==row['maximum_centroid_error'] and row['seconds']>0
            if row['name'] not in application_first:application_first[row['name']]=actual
            else:assert application_first[row['name']]==actual,'Application repeat changed'
    # Compare every saved candidate tensor with the matching rebuilt baseline.
    for (key,name,model,pass_index),actual in outputs.items():
        if key=='0-baseline':continue
        wanted=outputs[('0-baseline',name,model,pass_index)]
        a=next(r for r in results[key]['rows'] if (r['name'],r['model'],r['pass'])==(name,model,pass_index))
        b=next(r for r in results['0-baseline']['rows'] if (r['name'],r['model'],r['pass'])==(name,model,pass_index))
        assert a['input']==b['input'],'Different graph input across workers'
        check=scaled(actual,wanted)
        comparisons.append(dict(worker=key,name=name,model=model,pass_index=pass_index,reference='rebuilt-baseline',
            bit_identical=actual.tobytes()==wanted.tobytes(),**check))
    # Qualify the rebuilt baseline against already-closed current-product arrays.
    old=ROOT/'artifacts/pyannote-performance-profile-20260921/output'
    old_closed=old.parent/'closed.json'
    assert pin(old_closed)['sha256']=='4b7542a8917e558c92ba632fada6830ddf8c870fccf49064dae17ff1de194d5c'
    old_pins=json.loads(old_closed.read_text())['files']
    old_name=str((old/'result.json').relative_to(ROOT))
    assert pin(old/'result.json')==old_pins[old_name]
    native_files[old_name]=pin(old/'result.json');native_files[str(old_closed.relative_to(ROOT))]=pin(old_closed)
    old_result=json.loads((old/'result.json').read_text())
    for a,b in zip(results['0-baseline']['rows'],old_result['rows'],strict=True):
        assert a['input']['sha256']==b['input']['sha256'] and a['output']['sha256']==b['output']['sha256']
    timing=[]
    for case in manifest['cases']:
        name=case['name'];by_role={};workers=[]
        for key,result in results.items():
            observed=[r['seconds'] for r in result['applications'] if r['name']==name and r['phase']=='measured']
            assert len(observed)==3
            workers.append(dict(worker=key,seconds=observed,mean=statistics.mean(observed)))
            by_role.setdefault(key.split('-',1)[1],[]).extend(observed)
        baseline=statistics.mean(by_role['baseline']);candidate=statistics.mean(by_role['candidate'])
        timing.append(dict(name=name,baseline_seconds=baseline,candidate_seconds=candidate,ratio=candidate/baseline,
            reduction_percent=100*(1-candidate/baseline),workers=workers))
    passed=all(c['failed_values']==0 for c in comparisons)
    passed=passed and all(c.get('bit_identical',True) for c in comparisons)
    analysis=dict(passed=passed,comparisons=comparisons,profiles=profiles,timing=timing,output_arrays=arrays,
        output_values=sum(v.size for v in outputs.values()),public_calls=64,measured_public_calls=48,warmup_public_calls=16,
        samples=samples,peak_rss=max(r['peak_rss'] for r in state['runs']),identities=identities,
        limitation='Descriptive active-workstation comparison; no AMD speedup or statistical parity claim')
    save(BASE/'analysis.json',analysis)
    files=dict(spec['files']);files.update(native_files)
    for path in BASE.rglob('*'):
        if path.is_file() and not {'bin','obj'}.intersection(path.relative_to(BASE).parts):files[str(path.relative_to(ROOT))]=pin(path)
    for path in Path(__file__).parent.iterdir():
        if path.is_file():files[str(path.relative_to(ROOT))]=pin(path)
    save(BASE/'closed.json',dict(passed=passed,files=files,identities=identities))
    print(json.dumps(dict(passed=passed,arrays=arrays,values=analysis['output_values'],public_calls=64,
        max_native_error=max(c['max_scaled_error'] for c in comparisons if c['reference']=='native'),
        timing=timing,samples=samples,closure=pin(BASE/'closed.json'),files=len(files))))
    assert passed,'Numerical gate failed; complete diagnostics retained'


if __name__=='__main__':main()
