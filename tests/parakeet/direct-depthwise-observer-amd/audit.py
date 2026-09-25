"""Verify compiled observation scope, exact public records and every route count."""
import base64
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from run import BASE,REMOTE,ROOT,TOOLS,PRELUDE,APP,pin,read,write,ssh,prepared
from expected import audit_counts


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
    names=['sdk-version','core-restore','core-build','bridge-restore','bridge-build','inventory'] if kind=='build' else ['counts']
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
    built=read(folder/'built.json');assert built['passed'] and built['inventory']==pin(folder/'logs/instructions.json')
    assert built['runtime'].keys()==spec['runtime'].keys()
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted,name
    assert [n for n in built['runtime'] if built['runtime'][n]!=spec['runtime'][n]]==['Lokad.Onnx.dll']
    assert built['product']=={n:built['runtime'][n] for n in spec['before_product']}
    assert built['consumer']==spec['consumer']==built['runtime']['SampledAudio.dll']
    return folder,spec,receipt,state,built,resources


def check_inventory(inventory,spec,built):
    assert inventory['inventory_complete']
    assert [r['assembly'] for r in inventory['observations']]==['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']
    summaries=[]
    for r in inventory['observations']:
        core=r['assembly']=='Lokad.Onnx.dll'
        assert r['before_sha256']==spec['before_product'][r['assembly']]['sha256']
        assert r['after_sha256']==built['product'][r['assembly']]['sha256']
        assert r['methods']==(3281 if core else 697) and not r['removed']
        if core:
            assert len(r['differences'])==3 and {n.split('::')[1] for n in r['differences']}=={'Conv2DFloatCore','RunTiledBatchFloat','RunFloatMatMulKernel'}
            assert all(n.startswith('Lokad.Onnx.Tensor`1[T]::') for n in r['differences'])
            assert len(r['added'])==16 and all(n.startswith(('Lokad.Onnx.DepthwiseRouteProbe::','Lokad.Onnx.DepthwiseRouteProbe+')) for n in r['added'])
        else:assert not r['differences'] and not r['added']
        assert r['unchanged_methods']==r['methods']-(3 if core else 0)
        assert r['public_surface_equal'] and r['public_surface']==r['public_surface_after']
        assert r['assembly_attributes_before']==r['assembly_attributes_after']
        assert all(r['method_flags_after'][n]==v for n,v in r['method_flags_before'].items())
        assert set(r['candidate_methods'])==set(r['differences']+r['added'])
        summaries.append(dict(assembly=r['assembly'],unchanged=r['unchanged_methods'],changed=r['differences'],added=r['added']))
    return summaries


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    warnings=[]
    for run in state['runs']:
        text=(folder/'logs'/(run['name']+'.stdout')).read_text()+(folder/'logs'/(run['name']+'.stderr')).read_text()
        assert ': error ' not in text
        warnings.extend(line for line in text.splitlines() if ': warning ' in line)
    previous=read(ROOT/'artifacts/parakeet-packed-final-row-build-amd-20260925/build-review.json')
    allowed=[w.replace('/dev/shm/lokad-parakeet-packed-final-row-build-20260925',REMOTE) for w in previous['warnings']]
    assert Counter(warnings)==Counter(allowed),warnings
    scope=check_inventory(read(folder/'logs/instructions.json'),spec,built)
    result=dict(passed=True,built=pin(folder/'built.json'),products=built['product'],consumer=built['consumer'],scope=scope,
        consumer_unchanged=True,data_unchanged=True,arithmetic_source_reversal=True,source=spec['source'],warnings=warnings,
        resources=resources,diagnostic_only=True,release_admitted=False,reviewer=pin(Path(__file__)),collection=pin(folder/'build-collection.json'))
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
    write(BASE/'build-review-transferred.json',transferred);print(json.dumps(dict(**transferred,scope=scope)))


def module(name,path):
    loader=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(loader);loader.loader.exec_module(value);return value


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    run=state['runs'][0];accounting=module('route_accounting',folder/'campaign_processes.py')
    wanted=accounting.foreign_fraction(run['cpu_before'],run['cpu_after'],state['supervisor']['pid'])
    assert wanted==run['accounting'] and wanted['valid'] and wanted['foreign_cpu_fraction']<=.01
    protocol=module('route_protocol',folder/'protocol.py');manifest=read(APP/'collected/manifests/current-parakeet.json')
    value=read(folder/'probe/result.json');protocol.validate_records(value,manifest,'timing')
    assert value['passed'] and not value['sampled'] and value['runtime']=='.NET 10.0.8' and value['processor_count']==1
    assert value['core_sha256']==built['product']['Lokad.Onnx.dll']['sha256'] and value['runner_sha256']==built['consumer']['sha256']
    assert value['data_sha256']==spec['before_product']['Lokad.Onnx.Data.dll']['sha256']
    assert value['manifest_sha256']==pin(APP/'collected/manifests/current-parakeet.json')['sha256']
    reference=read(folder/'reference-public.json');expected={r['name']:r['result'] for r in reference['records']};assert len(expected)==20
    assert reference['core_sha256']==spec['reference_product']['Lokad.Onnx.dll']
    assert reference['data_sha256']==spec['reference_product']['Lokad.Onnx.Data.dll']
    assert len(value['records'])==80
    for i,row in enumerate(value['records']):
        assert row==read(folder/'probe'/f'{i:03}.json') and row['thread_id']==run['ready']['thread_id']
        assert row['result']==expected[row['name']]
    counts=read(folder/'logs/counts.json')
    assert counts['pid']==run['owner']['pid'] and counts['core_sha256']==built['product']['Lokad.Onnx.dll']['sha256']
    observed=audit_counts(counts,read(folder/'diagnosis.json'))
    analysis=dict(passed=True,observed=observed,resources=resources,products=built['product'],consumer=built['consumer'],
        source=spec['source'],diagnosis=spec['diagnosis'],public_requests=80,exact_public_results=True,
        diagnostic_only=True,release_admitted=False,instrumented_times_not_scored=True,accounting=wanted,
        raw_counts=pin(folder/'logs/counts.json'),public_result=pin(folder/'probe/result.json'))
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),
        collection=pin(folder/'capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),
        terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),observed={k:v for k,v in observed.items() if k!='buckets'},resources=resources)))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
