"""Qualify actual observer binaries before any M78 profile request starts."""
import base64
import json
from run import BASE,PRELUDE,prepared,pin,read,write,ssh
from compiled_scope import verify_data,verify_runner


def main():
    prepared();assert not (BASE/'build-review.json').exists()
    folder=BASE/'build-collected';spec=read(BASE/'bundle/spec.json')
    receipt=read(folder/'build-collection.json');transfer=read(BASE/'build-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'build-results.tar.gz')
    assert transfer['collection']==pin(folder/'build-collection.json') and receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'build-state.json')
    assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'build-deployment.json')
    assert [r['name'] for r in state['runs']]==['sdk-version','data-restore','data-build','bridge-restore','bridge-build','inventory']
    resources=[]
    for run in state['runs']:
        assert run['complete'] and run['code']==0
        limits=spec['build_limits']
        assert run['preflight']['available']>=limits['available_before'] and run['preflight']['tmpfs']>=limits['tmpfs_before']
        rows=[json.loads(line) for line in (folder/'logs'/(run['name']+'.resources.jsonl')).read_text().splitlines()]
        assert len(rows)==run['samples'] and rows and run['seconds']<limits['seconds']
        for row in rows:
            assert row['seconds']<limits['seconds'] and row['rss']<limits['rss']
            assert row['available']>=spec['minimum_free'] and row['tmpfs']>=spec['minimum_free'] and row['output']<spec['output_limit']
            assert row['rss']==sum(m['rss'] for m in row['members'])
            for member in row['members']:
                assert run['members'][str(member['pid'])]==member['birth']
                assert member['affinity']==[2] and all(t==[2] for t in member['threads'])
        resources.append(dict(name=run['name'],samples=len(rows),peak_rss=max(r['rss'] for r in rows)))
    built=read(folder/'built.json');assert built['core']==spec['core'] and built['consumer']==spec['original_consumer']
    for name,wanted in built['runtime_files'].items():assert pin(folder/name)==wanted,name
    for mode in ['runtime-control','runtime-observed']:
        for path in (BASE/'bundle/runtime-base').iterdir():
            if mode=='runtime-observed' and path.name=='Lokad.Onnx.Data.dll':continue
            assert pin(folder/mode/path.name)==pin(path),path.name
    assert pin(folder/'runtime-control/Lokad.Onnx.Data.dll')==spec['data']
    original=read(BASE/'bundle/evidence/original-observer-instructions.json')
    reference,=[r for r in original['observations'] if r['assembly']=='Lokad.Onnx.Data.dll']
    inventory=read(folder/'inventory/instructions.json');assert inventory['inventory_complete']
    assert [r['assembly'] for r in inventory['observations']]==['SampledAudio.dll','Lokad.Onnx.Data.dll']
    runner,data=inventory['observations']
    reviews=[verify_runner(runner,spec['original_consumer']['sha256']),
        verify_data(data,spec['data']['sha256'],built['data']['sha256'],reference)]
    warnings=[dict(log=path.name,text=line.strip()) for path in (folder/'logs').iterdir()
        if path.suffix in ['.stdout','.stderr'] for line in path.read_text(encoding='utf8').splitlines() if 'warning' in line.lower()]
    assert not warnings,warnings
    value=dict(passed=True,built=pin(folder/'built.json'),inventory=pin(folder/'inventory/instructions.json'),
        collection=pin(folder/'build-collection.json'),spec=pin(BASE/'bundle/spec.json'),
        core_unchanged=True,consumer_unchanged=True,constructor_unchanged=True,
        methods=reviews,resources=resources,warnings=warnings,reviewer=pin(__file__))
    write(BASE/'build-review.json',value)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    remote=ssh(PRELUDE+f'''
import base64
from remote import verify,pin,read,live
verify();state=read(base/'build-state.json');assert state['complete'] and state['code']==0
assert not live(state['supervisor']) and all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={value['built']!r} and pin(base/'inventory/instructions.json')=={value['inventory']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert remote['review']==pin(BASE/'build-review.json');write(BASE/'build-review-transferred.json',remote)
    print(json.dumps(dict(passed=True,review=remote['review'],methods=reviews,core=built['core'],data=built['data'],consumer=built['consumer'])))


if __name__=='__main__':main()
