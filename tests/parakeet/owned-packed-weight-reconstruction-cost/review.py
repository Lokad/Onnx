"""Independently reconcile complete reconstruction timing, contracts and resource observations."""
import base64
import csv
import json
from pathlib import Path
import sys
from run import BASE,ROOT,TOOLS,PRELUDE,JOBS,pin,read,write,ssh,prepared
from source_scope import verify as verify_source
from measurements import qualify,summarize


def collected(kind):
    prepared();folder=BASE/(kind+'-collected');spec=read(BASE/'bundle/spec.json')
    receipt=read(folder/(kind+'-collection.json'));transfer=read(BASE/(kind+'-transfer.json'))
    assert transfer['passed'] and transfer['archive']==pin(BASE/(kind+'-results.tar.gz'))
    assert transfer['collection']==pin(folder/(kind+'-collection.json'))
    assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    for name,wanted in spec['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/(kind+'-state.json'))
    assert state['complete'] and state['code']==0 and receipt['state']==pin(folder/(kind+'-state.json'))
    assert state['supervisor']==read(BASE/(kind+'-deployment.json'))
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    expected=['sdk-version','consumer-restore','consumer-build'] if kind=='build' else JOBS
    assert [r['name'] for r in state['runs']]==expected
    resources=[];limits=spec[kind+'_limits']
    for run in state['runs']:
        assert run['complete'] and run['code']==0 and run['seconds']<limits['seconds']
        assert run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
        samples=[json.loads(s) for s in (folder/'logs'/(run['name']+'.resources.jsonl')).read_text().splitlines()]
        assert len(samples)==run['samples']>0
        for sample in samples:
            assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss']
            assert min(sample['available'],sample['tmpfs'])>=spec['minimum_free'] and sample['output']<spec['output_limit']
            assert sample['rss']==sum(m['rss'] for m in sample['members'])
            for member in sample['members']:
                assert run['members'][str(member['pid'])]==member['birth'] and member['affinity']==[2]
                assert member['threads'] and all(t==[2] for t in member['threads'])
        gaps=[samples[0]['seconds']]+[b['seconds']-a['seconds'] for a,b in zip(samples,samples[1:])]+[run['seconds']-samples[-1]['seconds']]
        assert all(0<=g<10 for g in gaps)
        resources.append(dict(name=run['name'],samples=len(samples),seconds=run['seconds'],peak_rss=max(s['rss'] for s in samples)))
    built=read(folder/'built.json');assert built['passed'] and not built['product_rebuilt']
    assert set(built['runtime'])=={role+'/'+n for role,files in spec['runtime_files'].items() for n in [*files,'OwnedWeightCounters.dll','OwnedWeightCounters.deps.json','OwnedWeightCounters.runtimeconfig.json']}
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted,name
    for role,files in spec['runtime_files'].items():
        for name,wanted in files.items():assert built['runtime'][role+'/'+name]==wanted
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        assert built['runtime']['selected/OwnedWeightCounters.'+suffix]==built['runtime']['candidate/OwnedWeightCounters.'+suffix]
    return folder,spec,receipt,state,built,resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for name in ['consumer-restore','consumer-build']:
        text=(folder/'logs'/(name+'.stdout')).read_text()+(folder/'logs'/(name+'.stderr')).read_text()
        assert ': warning ' not in text and ': error ' not in text
    assert '0 Warning(s)' in text and '0 Error(s)' in text
    source=(folder/'source/Program.cs').read_text()
    assert 'GC.Collect(' not in source and 'new ParakeetTranscriber(' in source and 'OnnxImport.Load' not in source
    assert 'Profiler.BeginExecution(true)' in source and 'enc.LastCopyBytes' in source and 'enc.LastScratchBytes' in source
    assert 's.Time.Ticks' in source and 'owner.Transcribe(' not in source
    assert verify_source((folder/'evidence/original-counter.cs.txt').read_text(),source)==spec['diagnostic_evidence']['source_review']
    project=(folder/'source/OwnedWeightCounters.csproj').read_text();assert '<ProjectReference' not in project and project.count('<Reference ')==6
    result=dict(passed=True,product_rebuilt=False,zero_added_warnings=True,built=pin(folder/'built.json'),
        collection=pin(folder/'build-collection.json'),transfer=pin(BASE/'build-transfer.json'),source=pin(folder/'source/Program.cs'),
        resources=resources,products=spec['products'],model_closure=spec['model_closure'],census_closure=spec['census_closure'],
        release_admitted=False,reviewer=pin(Path(__file__)))
    write(BASE/'build-review.json',result)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    transferred=ssh(PRELUDE+f'''
from remote import verify,live,read,pin
import base64
import csv
verify();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r}
with (base/'build-review.json').open('xb') as f:f.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert transferred['review']==pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',transferred)
    print(json.dumps(dict(passed=True,review=transferred['review'],resources=resources)))


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    assert spec['jobs']==JOBS==['selected-512-a','candidate-512-a','candidate-512-b','selected-512-b']
    assert spec['passes']==4 and spec['warmup_passes']==1 and not spec['release_admitted']
    clocks={};results={};raw=[]
    for name in JOBS:
        role,mode,leg=name.split('-');assert mode=='512' and leg in ['a','b']
        wait=read(folder/(name+'-preflight-wait.json'))
        assert wait and all(0<=row['seconds']<spec['preflight_wait_seconds'] for row in wait)
        assert wait[-1]['available']>=spec['capture_limits']['available_before']
        assert all(row['tmpfs']>=spec['capture_limits']['tmpfs_before'] for row in wait)
        path=folder/'probe'/name
        result=read(path/'result.json')
        assert result['spec_sha256']==pin(folder/'spec.json')['sha256']
        assert {p.name for p in path.iterdir()}=={'result.json'}|{f'{i:03}.json' for i in range(80)}
        for index,record in enumerate(result['records']):assert record==read(path/f'{index:03}.json')
        clocks[name]=qualify(result,spec,role,built['runtime'][role+'/OwnedWeightCounters.dll'])
        results[name]=dict(passed=True,result=pin(path/'result.json'),requests=80,
                          frontend_encoder_exact=True,actual_traffic_exact=True)
        raw.extend(dict(process=name,**row) for row in clocks[name])
    assert len(raw)==320 and sum(row['phase']=='measured' for row in raw)==240
    assert sum(row['reconstructions'] for row in raw)==4872
    for left,right in [('selected-512-a','candidate-512-a'),('selected-512-b','candidate-512-b')]:
        for a,b in zip(clocks[left],clocks[right],strict=True):
            assert (a['name'],a['pass_index'],a['phase'],a['frames'])==(b['name'],b['pass_index'],b['phase'],b['frames'])
            assert a['scratch_bytes']-b['scratch_bytes']==1459617792
            assert b['copy_bytes']-a['copy_bytes']==(1459617792 if b['reconstructions'] else 0)
    summary=summarize(clocks,spec['cases'])
    analysis=dict(passed=True,diagnostic_only=True,products=spec['products'],
        application_closure=spec['diagnostic_evidence']['application_closure'],
        counter_closure=spec['diagnostic_evidence']['counter_closure'],
        model_closure=spec['model_closure'],census_closure=spec['census_closure'],
        source_review=spec['diagnostic_evidence']['source_review'],consumer=built['runtime']['candidate/OwnedWeightCounters.dll'],
        results=results,requests=320,measured=240,warmup=80,feed_forward_calls=30720,resources=resources,
        summary=summary,application_scored=False,release_admitted=False,new_variant_selected=False,
        prior_failed_application_gate=spec['diagnostic_evidence']['failed_application_gate'],
        failed_release_controls=spec['failed_release_controls'])
    with (BASE/'clocks.csv').open('x',encoding='utf8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(raw[0]));writer.writeheader();writer.writerows(raw)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,usable_for_attribution=summary['usable_for_attribution'],
        analysis=pin(BASE/'analysis.json'),collection=pin(folder/'capture-collection.json'),
        transfer=pin(BASE/'capture-transfer.json'),terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    corpus=summary['table'][-1]
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),usable_for_attribution=summary['usable_for_attribution'],
        controls=sum(row['passed'] for row in summary['controls']),failed_controls=[row for row in summary['controls'] if not row['passed']],
        corpus={role:{metric:value['seconds'] for metric,value in corpus[role].items()} for role in ['selected','candidate']},
        resources=resources),indent=2))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
