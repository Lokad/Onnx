"""Recompute numerical, ownership, profile and allocation evidence."""
from common import *
import re
import statistics
import numpy as np


def main():
    spec=read(BASE/'manifest.json');verify(spec)
    state=read(BASE/'processes.json');qualification=read(BASE/'qualification.json');corrected=read(BASE/'qualification-cli.json')
    assert state['complete'] and state['code']==0 and qualification['complete'] and qualification['code']==1
    assert corrected['complete'] and corrected['code']==1 and corrected['predecessor']==pin(BASE/'qualification.json')
    serial=read(BASE/'qualification-serial.json')
    assert serial['complete'] and serial['code']==0 and serial['predecessor']==pin(BASE/'qualification-cli.json')
    assert absent(state['supervisor']) and [r['role'] for r in state['runs']]==JOBS
    assert [r['name'] for r in qualification['stages']]==['il-build','baseline-equivalence','candidate-difference','backend-build','backend-tests']
    assert all(r['code']==0 for r in qualification['stages'][:-1]) and qualification['stages'][-1]['code']==1
    assert [(r['name'],r['code']) for r in corrected['stages']]==[('cli-build',0),('backend-with-cli-tests',1)]
    assert [r['name'] for r in serial['stages']]==['import-diagnostic-isolated','backend-serial-tests','tensors-build','tensors-tests']
    assert all(r['code']==0 for r in serial['stages'])
    config=BASE/'candidate-source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0/xunit.runner.json'
    assert pin(config)==serial['configuration'] and read(config)==dict(parallelizeTestCollections=False,maxParallelThreads=1)
    for name,passed,skipped in [('backend-serial-tests',3101,93),('tensors-tests',342,0)]:
        log=(BASE/'logs'/(name+'.stdout')).read_text()
        match=re.search(r'Passed!\s+- Failed:\s+0, Passed:\s+(\d+), Skipped:\s+(\d+), Total:\s+(\d+)',log)
        assert match and tuple(map(int,match.groups()))==(passed,skipped,passed+skipped),name
    identities=[state['supervisor']];samples=0
    for run in state['runs']:
        assert run['complete'] and run['code']==0 and absent(run['worker'])
        identities.append(run['worker'])
        assert run['preflight']['available']>=LIMITS['preflight_available'] and run['preflight']['disk']>=LIMITS['disk']
        rows=[json.loads(s) for s in (BASE/'process'/run['role']/'samples.jsonl').read_text().splitlines()]
        assert len(rows)==run['samples']>0 and max(s['rss'] for s in rows)==run['peak_rss']
        for s in rows:
            assert s['seconds']<LIMITS['seconds'] and s['rss']<LIMITS['rss'] and s['available']>=LIMITS['available']
            assert s['disk']>=LIMITS['disk'] and s['affinity']==[2] and s['pid']==run['worker']['pid'] and s['birth']==run['worker']['birth']
        samples+=len(rows)
    baseline_il=read(BASE/'baseline-equivalence.json');candidate_il=read(BASE/'candidate-difference.json')
    old=baseline_il['observations'][0];new=candidate_il['observations'][0]
    assert old['equal'] and not old['differences'] and old['before_sha256']=='d1f86a7346dcd70ebcc9ef7d9cd9633f05ad3a5275ca39f035c72325a0531fa4'
    assert old['after_sha256']==new['before_sha256']==spec['cores']['baseline']['sha256']
    assert new['after_sha256']==spec['cores']['candidate']['sha256'] and len(new['differences'])==2
    assert all('RunCoreInner' in name for name in new['differences'])
    for record in (old,new):
        before,after=record['before_methods'],record['after_methods']
        assert before.keys()==after.keys() and len(before)==record['methods']
        assert [k for k in before if before[k]!=after[k]]==record['differences']
    instructions=read(BASE/'outputs/candidate/il.json')['instructions']
    enabled=next(i for i,r in enumerate(instructions) if r['operand']=='Lokad.Onnx.ProfilerContext::Boolean Enabled')
    wall=next(i for i,r in enumerate(instructions) if r['operand']=='Lokad.Onnx.ProfilerContext::Boolean get_WallOnly()')
    ebranch,wbranch=instructions[enabled+1],instructions[wall+1]
    assert ebranch['opcode'].startswith('brfalse') and wbranch['opcode'].startswith('brtrue') and ebranch['operand']==wbranch['operand']
    captured=[r for r in instructions if r['opcode']=='newobj' and ('DisplayClass189_0' in r['operand'] or 'Func`1[System.String]' in r['operand'])]
    assert len(captured)==2 and all(wbranch['offset']<r['offset']<wbranch['operand'] for r in captured)
    # No branches enter the skipped interval from outside the detailed path.
    first,last=captured[0]['offset'],wbranch['operand']
    for r in instructions:
        if isinstance(r['operand'],int) and r['opcode'].startswith(('br','beq','bne','bge','bgt','ble','blt','leave')):
            if first<=r['operand']<last:assert first<=r['offset']<last
    results={};groups={};metrics=[];full_arrays=0
    expected_keys=[name+'-'+policy+'-'+mode for name in CASES for policy in ('default','memory') for mode in ('disabled','detailed','wall')]
    for role in JOBS:
        folder=BASE/'outputs'/role;r=read(folder/'result.json');results[role]=r
        assert r['complete'] and r['role']==role and r['manifest_sha256']==pin(BASE/'manifest.json')['sha256']
        assert r['core_sha256']==spec['cores'][role]['sha256'] and r['runtime']=='10.0.12' and r['affinity']==[2] and r['processor_count']==1
        assert r['inputs_unchanged'] and r['held_outputs_unchanged'] and not r['native_loaded']
        assert [g['key'] for g in r['groups']]==expected_keys and len(r['rows'])==960
        groups[role]={}
        for group in r['groups']:
            key=group['key'];rows=[row for row in r['rows'] if row['key']==key];groups[role][key]=rows
            assert [row['call'] for row in rows]==list(range(32)) and [row['warmup'] for row in rows]==[True]*16+[False]*16
            profile=(folder/key/'profile.json').read_bytes();profile_hash=hashlib.sha256(profile).hexdigest()
            for row in rows:
                assert row['nodes']==group['nodes'] and row['sha256']==group['output_sha256'] and row['profile_sha256']==profile_hash
                assert 0<=row['maximum_native_error']<=1e-4 and 0<=row['run_allocated']<=row['public_allocated']
            case=next(c for c in CASES if key.startswith(c+'-'));fixture=read(ROOT/spec['fixtures']/(case+'.json'))
            wanted=np.fromfile(ROOT/spec['fixtures']/fixture['reference_file'],dtype='<f4')
            for call in (0,31):
                path=folder/key/(str(call)+'.f32');assert pin(path)['sha256']==group['output_sha256']
                actual=np.fromfile(path,dtype='<f4');assert actual.shape==wanted.shape and np.isfinite(actual).all()
                maximum=float(np.max(np.abs(actual.astype(np.float64)-wanted)/np.maximum(1.,np.abs(wanted.astype(np.float64)))))
                assert maximum==rows[call]['maximum_native_error']<=1e-4
                full_arrays+=1
    assert results['baseline']['groups']==results['candidate']['groups']
    positive=True
    for key in expected_keys:
        b,c=groups['baseline'][key],groups['candidate'][key]
        for left,right in zip(b,c,strict=True):
            assert all(left[k]==right[k] for k in ('key','call','warmup','nodes','sha256','maximum_native_error','profile_sha256'))
        for boundary in ('run_allocated','public_allocated'):
            before=[r[boundary] for r in b[16:]];after=[r[boundary] for r in c[16:]]
            delta=[x-y for x,y in zip(before,after,strict=True)]
            mode=key.rsplit('-',1)[1]
            if mode!='detailed':positive=positive and min(delta)>0
            metrics.append(dict(key=key,boundary=boundary,nodes=b[0]['nodes'],baseline_min=min(before),baseline_max=max(before),
                baseline_median=statistics.median(before),candidate_min=min(after),candidate_max=max(after),
                candidate_median=statistics.median(after),saving_min=min(delta),saving_max=max(delta),
                saving_median=statistics.median(delta),saving_per_node_median=statistics.median(delta)/b[0]['nodes']))
    summary=dict(qualified=positive,correctness=True,allocation_screen=positive,metrics=metrics,calls=1920,
        full_arrays=full_arrays,maximum_native_error=max(r['maximum_native_error'] for result in results.values() for r in result['rows']),
        original_core_methods=old['methods'],unchanged_candidate_methods=new['methods']-len(new['differences']),
        changed_methods=new['differences'],guarded_allocation_offsets=[r['offset'] for r in captured],
        skip_target=wbranch['operand'],samples=samples,peak_rss=max(r['peak_rss'] for r in state['runs']),identities=identities)
    write(BASE/'analysis.json',summary)
    files=dict(spec['files'])
    for path in BASE.rglob('*'):
        if path.is_file() and not any(part in ('obj',) for part in path.relative_to(BASE).parts):files[rel(path)]=pin(path)
    for path in TOOLS.iterdir():
        if path.is_file():files[rel(path)]=pin(path)
    write(BASE/'closed.json',dict(qualified=positive,files=files,identities=identities,analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(qualified=positive,calls=1920,full_arrays=full_arrays,samples=samples,
        disabled=[m for m in metrics if m['key'].endswith('-disabled') and m['boundary']=='run_allocated'],closure=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
