"""Check every original clock, product identity, owner, bound and fixed gate."""
import base64
import importlib.util
import json
from pathlib import Path
import sys
from run import BASE,REMOTE,TOOLS,PRELUDE,pin,read,write,ssh,prepared
from score import ORDER,score


def collected(kind):
    prepared();folder=BASE/(kind+'-collected');spec=read(BASE/'bundle/spec.json')
    receipt=read(folder/(kind+'-collection.json'));transfer=read(BASE/(kind+'-transfer.json'))
    assert transfer['passed'] and transfer['archive']==pin(BASE/(kind+'-results.tar.gz'))
    assert transfer['collection']==pin(folder/(kind+'-collection.json')) and receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    for name,wanted in spec['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/(kind+'-state.json'))
    assert state['complete'] and state['code']==0 and receipt['state']==pin(folder/(kind+'-state.json'))
    assert state['supervisor']==read(BASE/(kind+'-deployment.json'))
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    names=['sdk-version','consumer-restore','consumer-build'] if kind=='build' else ORDER
    assert [r['name'] for r in state['runs']]==names
    limits=spec[kind+'_limits'];resources=[]
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
    built=read(folder/'built.json');assert built['passed'] and built['products']==spec['products']
    for name,wanted in built['runtime_files'].items():assert pin(folder/name)==wanted,name
    runtimes={role:{p.relative_to(folder/'runtime'/role).as_posix():pin(p) for p in (folder/'runtime'/role).rglob('*') if p.is_file()} for role in ['current','candidate']}
    assert set(runtimes['current'])==set(runtimes['candidate'])
    assert {n for n in runtimes['current'] if runtimes['current'][n]!=runtimes['candidate'][n]}=={'Lokad.Onnx.dll'}
    for role,files in runtimes.items():
        assert files['Lokad.Onnx.dll']==spec['products'][role] and files['Screen.dll']==built['consumer']
        assert files['Google.Protobuf.dll']==spec['external'][spec['runtimes']['current']+'/Google.Protobuf.dll']
    return folder,spec,receipt,state,built,resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    for run in state['runs']:
        text=(folder/'logs'/(run['name']+'.stdout')).read_text()+(folder/'logs'/(run['name']+'.stderr')).read_text()
        assert ': error ' not in text and ': warning ' not in text,text
    result=dict(passed=True,built=pin(folder/'built.json'),products=built['products'],consumer=built['consumer'],
        same_consumer=True,only_runtime_difference='Lokad.Onnx.dll',warnings=[],resources=resources,
        evidence=spec['evidence'],reviewer=pin(Path(__file__)),collection=pin(folder/'build-collection.json'))
    write(BASE/'build-review.json',result)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    transferred=ssh(PRELUDE+f'''
from remote import verify,live,read,pin
import base64
verify();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r}
with (base/'build-review.json').open('xb') as f:f.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert transferred['review']==pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',transferred);print(json.dumps(transferred))


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    module=importlib.util.spec_from_file_location('screen_accounting',folder/'campaign_processes.py')
    accounting=importlib.util.module_from_spec(module);module.loader.exec_module(accounting)
    reports={}
    for sequence,run in enumerate(state['runs']):
        name=run['name'];role=name.split('-')[0]
        wanted=accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
        assert wanted==run['accounting'] and wanted['valid'] and wanted['foreign_cpu_fraction']<=.01
        value=read(folder/'logs'/(name+'.json'))
        assert value['pid']==run['owner']['pid'] and value['runtime']=='10.0.8' and not value['flags']
        assert value['core_sha256']==spec['products'][role]['sha256'] and value['assembly']==built['consumer']['sha256']
        assert value['census_sha256']==spec['census']['sha256']
        assert run['command']==['/home/vermorel/.dotnet/dotnet',REMOTE+'/runtime/'+role+'/Screen.dll',REMOTE,role,str(sequence),REMOTE+'/logs/'+name+'.json']
        reports[name]=value
    verdict=score(reports,read(folder/'census.json'))
    analysis=dict(passed=True,**verdict,resources=resources,products=spec['products'],consumer=built['consumer'],
        evidence=spec['evidence'],diagnostic_only=True,release_admitted=False,failed_graph_cases=spec['failed_graph_cases'],
        no_model_execution=True,no_application_score=True,accounting=[r['accounting'] for r in state['runs']],
        outputs={name:pin(folder/'logs'/(name+'.json')) for name in ORDER})
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,admitted=analysis['admitted'],analysis=pin(BASE/'analysis.json'),
        collection=pin(folder/'capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),
        terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),admitted=analysis['admitted'],weighted=analysis['corpus_weighted'],
        failed_controls=[r for r in analysis['controls'] if not r['passed']],failed_cases=[r for r in analysis['rows'] if not r['passed']],
        gates=analysis['gates'],resources=resources)))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
