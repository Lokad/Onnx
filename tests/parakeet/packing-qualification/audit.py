"""Compare complete candidate trajectories with independently qualified original bytes."""
import math
from qualify import ROOT,TOOLS,BASE,ADMISSION,TRACE,MANIFEST,read,pin,verify,save,terminal


def main():
    prepared=read(BASE/'prepared.json');assert prepared['passed'];verify(prepared['files'])
    admission=read(ADMISSION/'prepared.json');assert admission['passed'];verify(admission['files'])
    closed=read(TRACE/'closed.json');assert closed['passed'];verify(closed['files'])
    state=read(BASE/'processes.json');assert state['complete'] and state['code']==0;terminal(state['supervisor'])
    assert [r['mode'] for r in state['runs']]==['trace','public']
    for run in state['runs']:
        assert run['complete'] and run['passed'] and run['code']==0 and run['samples']>0;terminal(run['worker'])
        samples=[__import__('json').loads(line) for line in (BASE/'logs'/(run['mode']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==run['samples'] and max(s['rss'] for s in samples)==run['peak_rss']
        assert all(s['seconds']<1200 and s['rss']<12*1024**3 and s['available']>=1024**3 and s['disk']>=20*1024**3 and s['affinity']==[2] for s in samples)
    spec=read(MANIFEST);original=TRACE/'trace-output';candidate=BASE/'trace-output'
    old=read(original/'result.json');new=read(candidate/'result.json');public=read(BASE/'public-output/result.json')
    assert old['passed'] and new['passed'] and public['passed']
    for item in (new,public):
        assert item['error'] is None and item['inputs_and_held_outputs_unchanged']
        assert item['flags']=={} and item['affinity']==4 and item['processor_count']==1 and item['runtime']=='.NET 10.0.12'
        assert item['manifest_sha256']==pin(MANIFEST)['sha256']
        for key,name in [('core_sha256','Lokad.Onnx.dll'),('data_sha256','Lokad.Onnx.Data.dll'),('runner_sha256','Profile.dll')]:assert item[key]==pin(BASE/'bin'/name)['sha256']
    assert new['mode']=='trace' and public['mode']=='public' and public['call_files']==[] and new['applications']==[]
    assert new['call_files']==[f'{i:04}.json' for i in range(1240)]==old['call_files'][:1240]
    assert read(candidate/'graphs.json')==read(original/'graphs.json')==read(BASE/'public-output/graphs.json')
    assert len(new['traced'])==len(public['applications'])==len(spec['cases'])==20
    for traced,application,c in zip(new['traced'],public['applications'],spec['cases'],strict=True):
        assert traced['name']==application['name']==c['name'] and traced['pass']==0
        assert traced['result']==application['result']==c['expected']
    coverage=[(c['name'],g,s) for c in spec['cases'] for g,s in [('frontend',-1),('encoder',-1)]+[('decoder',i) for i in range(c['expected']['decoder_calls'])]]
    counters=dict(graph_calls=0,input_arrays=0,output_arrays=0,values=0);seen=set()
    for filename,wanted in zip(new['call_files'],coverage,strict=True):
        a=read(original/filename);b=read(candidate/filename)
        assert (b['name'],b['graph'],b['step'])==wanted and b['pass']==a['pass']==0 and b['nodes']==[]
        assert b['frequency']>0 and b['reset_start_ticks']<=b['reset_end_ticks']<=b['start_ticks']<b['end_ticks']
        for group,counter in [('inputs','input_arrays'),('outputs','output_arrays')]:
            assert a[group].keys()==b[group].keys()
            for key,got in b[group].items():
                reference=a[group][key]
                for field in ('dtype','shape','values','sha256'):assert got[field]==reference[field],(filename,group,key,field)
                assert got['dtype'] in ('<f4','<i4','<i8') and got['values']==math.prod(got['shape'])
                path=(candidate/got['file']).resolve();assert path.is_relative_to((candidate/'arrays').resolve()) and got['file'] not in seen;seen.add(got['file'])
                actual=pin(path);assert actual['sha256']==got['sha256'] and actual['bytes']==got['values']*(8 if got['dtype']=='<i8' else 4)
                assert actual==pin(original/reference['file'])
                counters[counter]+=1;counters['values']+=got['values']
        counters['graph_calls']+=1
    assert counters['graph_calls']==1240 and counters['input_arrays']==6080 and counters['output_arrays']==4880
    packing=read(candidate/'packing.json');assert packing==read(BASE/'public-output/packing.json')
    for name,graph in packing.items():
        assert graph['maximum_packed_bytes']==dict(frontend=0,encoder=256*1024**2,decoder=64*1024**2)[name]
        assert sum(w['bytes'] for w in graph['weights'])==graph['retained_packed_bytes']<=graph['maximum_packed_bytes']
        assert all(w['bytes']==math.prod(w['shape'])*4 and w['shape'][0]<=4096 for w in graph['weights'])
    assert any(w['shape'][0]==4096 for w in packing['encoder']['weights'])
    files=dict(prepared['files'])
    for p in BASE.rglob('*'):
        if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts):files[p.relative_to(ROOT).as_posix()]=pin(p)
    for p in TOOLS.iterdir():
        if p.is_file():files[p.relative_to(ROOT).as_posix()]=pin(p)
    assert not (BASE/'closed.json').exists()
    save(BASE/'closed.json',dict(passed=True,files=files,counts=counters,packing=packing,resources=state['runs'],
        scope='Complete managed tensor/public equivalence; original three Windows ORT duration-logit failures remain; no new timing or AMD qualification'))
    print(__import__('json').dumps(dict(passed=True,counts=counters,closed=pin(BASE/'closed.json'))))


if __name__=='__main__':main()
