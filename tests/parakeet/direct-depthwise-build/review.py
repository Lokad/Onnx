"""Check complete compiled scope and actual focused numerical results."""
import base64
from collections import Counter
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
from run import BASE,BEFORE,REMOTE,REMOTE_RUNTIME,TOOLS,PRELUDE,pin,read,write,ssh,prepared

ADDED={'TryConvDirectDepthwise','RunDirectDepthwiseLine','RunDirectDepthwiseSpatial','DirectDepthwisePoint'}


def check_inventory(inventory,spec,built):
    assert inventory['inventory_complete']
    assert [r['assembly'] for r in inventory['observations']]==['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']
    summaries=[]
    for row in inventory['observations']:
        name=row['assembly'];core=name=='Lokad.Onnx.dll'
        assert row['before_sha256']==spec['before_product'][name]['sha256']
        assert row['after_sha256']==built['product'][name]['sha256']
        assert row['methods']==(3277 if core else 697) and not row['removed']
        assert len(row['differences'])==(1 if core else 0)
        assert {k.split('::')[1] for k in row['differences']}==({'Conv2DFloatCore'} if core else set())
        assert len(row['added'])==(4 if core else 0)
        assert {k.split('::')[1] for k in row['added']}==(ADDED if core else set())
        assert all(k.startswith('Lokad.Onnx.Tensor`1[T]::') for k in row['differences']+row['added'])
        assert all(row['method_flags_after'][k]==v for k,v in row['method_flags_before'].items())
        assert all(row['method_flags_after'][k]==0 for k in row['added'])
        assert row['public_surface_equal'] and row['public_surface']==row['public_surface_after']
        assert row['assembly_attributes_before']==row['assembly_attributes_after']
        assert row['unchanged_methods']==row['methods']-(1 if core else 0)
        assert set(row['candidate_methods'])==set(row['differences']+row['added'])
        if not core:assert built['product'][name]==spec['before_product'][name]
        summaries.append(dict(assembly=name,methods=row['methods'],unchanged=row['unchanged_methods'],changed=row['differences'],added=row['added'],removed=[]))
    return summaries



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
    names=['sdk-version','backend-restore','backend-build','bridge-restore','bridge-build','inventory'] if kind=='build' else ['contracts-normal','contracts-scalar']
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
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted,name
    assert built['consumer']==built['runtime']['Lokad.Onnx.Backend.Tests.dll']
    assert built['product']=={n:built['runtime'][n] for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    return folder,spec,receipt,state,built,resources


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build')
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    warnings=[]
    for run in state['runs']:
        output=(folder/'logs'/(run['name']+'.stdout')).read_text()+(folder/'logs'/(run['name']+'.stderr')).read_text()
        assert ': error ' not in output
        warnings.extend(line for line in output.splitlines() if ': warning ' in line)
    baseline=read(BEFORE/'build-review.json')
    allowed=[w.replace('/dev/shm/lokad-parakeet-packed-final-row-build-20260925',REMOTE) for w in baseline['warnings']]
    assert Counter(warnings)==Counter(allowed),warnings
    inventory=read(folder/'logs/instructions.json');methods=check_inventory(inventory,spec,built)
    result=dict(passed=True,built=pin(folder/'built.json'),inventory=pin(folder/'logs/instructions.json'),
        product=built['product'],consumer=built['consumer'],source=spec['source_prepared'],methods=methods,
        warnings=warnings,zero_added_warnings=True,resources=resources,public_surface_unchanged=True,
        data_methods_unchanged=True,shared_matrix_methods_unchanged=True,data_binary_unchanged=True,release_admitted=False,failed_graph_cases=spec['failed_graph_cases'],
        reviewer=pin(Path(__file__)),collection=pin(folder/'build-collection.json'))
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
    write(BASE/'build-review-transferred.json',transferred)
    print(json.dumps(dict(review=transferred['review'],methods=methods,product=built['product'])))


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    suites=[]
    for mode,count in spec['expected_tests'].items():
        path=folder/'logs'/('depthwise-'+mode+'.trx');root=ET.parse(path).getroot();counters=root.find('.//{*}Counters')
        assert counters is not None
        assert all(int(counters.get(k,'-1'))==count for k in ['total','executed','passed'])
        assert all(int(counters.get(k,'-1'))==0 for k in ['failed','error','timeout','aborted','notExecuted','notRunnable'])
        results=root.findall('.//{*}UnitTestResult');assert len(results)==count
        assert all(r.get('outcome')=='Passed' for r in results)
        names=[r.get('testName') for r in results];assert len(set(names))==count
        assert all('DirectDepthwiseTests.' in n for n in names)
        geometry=read(folder/'logs'/('geometry-'+mode+'.json'))
        census=read(folder/'census.json')['rows']
        assert geometry['passed'] and geometry['geometries']==59 and len(geometry['records'])==59
        assert geometry['original_core_sha256']==spec['before_product']['Lokad.Onnx.dll']['sha256']
        assert [r['geometry'] for r in geometry['records']]==[r['geometry'] for r in census]
        for row in geometry['records']:
            g=row['geometry'];assert row['bitwise_equal'] and row['checked_values']==g[0]*g[4]*g[15]*g[16]
        assert geometry['checked_values']==sum(r['checked_values'] for r in geometry['records'])
        suites.append(dict(mode=mode,passed=count,skipped=0,trx=pin(path),names=names,geometry=geometry))

    analysis=dict(passed=True,compiled_review=pin(BASE/'build-review.json'),product=built['product'],consumer=built['consumer'],
        source=spec['source_prepared'],suites=suites,resources=resources,release_admitted=False,failed_graph_cases=spec['failed_graph_cases'],
        no_model_execution=True,no_application_score=True,numerical_full_models_pending=True,application_comparison_pending=True)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),compiled_review=pin(BASE/'build-review.json'),
        collection=pin(folder/'capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),
        terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),suites=[dict(mode=s['mode'],passed=s['passed'],geometries=s['geometry']['geometries'],values=s['geometry']['checked_values']) for s in suites],product=built['product'],resources=resources)))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
