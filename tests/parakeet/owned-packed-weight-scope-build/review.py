"""Require only the diagnosed preparation predicate change before execution."""
import base64
import json
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET
from run import BASE,ROOT,TOOLS,PRELUDE,pin,read,write,ssh,prepared,source_verified


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
    expected=['sdk-version','backend-restore','backend-build','bridge-restore','bridge-build','inventory'] if kind=='build' else ['contracts-512','contracts-256','contracts-scalar']
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
    built=read(folder/'built.json');assert built['passed'] and built['inventory']==pin(folder/'logs/instructions.json')
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted,name
    assert built['consumer']==built['runtime']['Lokad.Onnx.Backend.Tests.dll']
    assert built['product']=={n:built['runtime'][n] for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
    return folder,spec,receipt,state,built,resources


def method(key):
    owner,name,_=key.split('::',2)
    return owner,name


def build():
    assert not (BASE/'build-review.json').exists()
    folder,spec,receipt,state,built,resources=collected('build');source=source_verified()
    assert (folder/'logs/sdk-version.stdout').read_text().strip()=='10.0.204'
    warnings=[]
    for run in state['runs']:
        output=(folder/'logs'/(run['name']+'.stdout')).read_text()+(folder/'logs'/(run['name']+'.stderr')).read_text()
        assert ': error ' not in output
        for line in output.splitlines():
            if ': warning ' in line:
                assert run['name']=='backend-build' and re.search(r'Zzz\.WideProjectionEntry\.cs\(20,(29|32)\): warning CS8604:',line),line
                warnings.append(line)
    assert len(warnings)==4,'Only the two established Core diagnostics, repeated in the build summary'
    foundation=(TOOLS.parent/'slice-dense-conversion-build-amd/Bridge.cs.txt').read_bytes().split(b'var observations = new List<object>();')[0]
    assert (folder/'bridge-source/Program.cs').read_bytes().startswith(foundation)
    inventory=read(folder/'logs/instructions.json');assert inventory['inventory_complete']
    rows=inventory['observations'];assert [r['assembly'] for r in rows]==['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']
    expected_core={('Lokad.Onnx.ComputationalGraph','PrepareOwnedMatMulWeights')}
    expected_data=set()
    summaries=[]
    for row,baseline in zip(rows,source['inventory']['observations']):
        name=row['assembly'];assert name==baseline['assembly']
        assert row['before_sha256']==baseline['before_sha256']==spec['before_product'][name]['sha256']
        assert row['after_sha256']==built['product'][name]['sha256']
        assert row['normalized_methods']==baseline['normalized_methods']
        assert row['method_flags_before']==baseline['method_flags_before']
        assert row['public_surface']==baseline['public_surface']
        assert row['public_surface_equal'] and row['public_surface_after']==row['public_surface']
        assert not row['removed']
        assert {method(k) for k in row['differences']}==(expected_core if name=='Lokad.Onnx.dll' else expected_data),row['differences']
        assert len(row['differences'])==(1 if name=='Lokad.Onnx.dll' else 0)
        assert all(row['method_flags_after'][k]==v for k,v in row['method_flags_before'].items())
        before=set(row['assembly_attributes_before']);after=set(row['assembly_attributes_after'])
        assert not before-after
        added_attributes=after-before
        assert not row['added'] and not added_attributes
        assert set(row['candidate_methods'])==set(row['differences'])|set(row['added'])
        assert row['unchanged_methods']==row['methods']-len(row['differences'])
        summaries.append(dict(assembly=name,original_methods=row['methods'],unchanged=row['unchanged_methods'],
            differences=row['differences'],added=row['added'],added_attributes=sorted(added_attributes)))
    result=dict(passed=True,zero_added_warnings=True,existing_warning_occurrences=len(warnings),warnings=warnings,
        built=pin(folder/'built.json'),inventory=pin(folder/'logs/instructions.json'),source=spec['source_prepared'],
        product=built['product'],consumer=built['consumer'],methods=summaries,resources=resources,
        public_surface_unchanged=True,existing_arithmetic_unchanged=True,release_admitted=False,
        failed_release_controls=spec['failed_release_controls'],collection=pin(folder/'build-collection.json'),reviewer=pin(Path(__file__)))
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
    print(json.dumps(dict(passed=True,review=transferred['review'],methods=summaries,product=built['product'],resources=resources)))


def capture():
    assert not (BASE/'closed.json').exists()
    folder,spec,receipt,state,built,resources=collected('capture')
    assert pin(folder/'build-review.json')==pin(BASE/'build-review.json')
    assert read(BASE/'build-review.json')['built']==pin(folder/'built.json')
    assert read(BASE/'build-collected/build-state.json')['ended']<state['started']
    suites=[]
    for mode,count in spec['expected_tests'].items():
        path=folder/'logs'/('owned-'+mode+'.trx');root=ET.parse(path).getroot()
        counters=root.find('.//{*}Counters');assert counters is not None
        assert all(int(counters.get(k,'-1'))==count for k in ['total','executed','passed'])
        assert all(int(counters.get(k,'-1'))==0 for k in ['failed','error','timeout','aborted','notExecuted','notRunnable'])
        results=root.findall('.//{*}UnitTestResult');assert len(results)==count
        assert all(r.get('outcome')=='Passed' for r in results)
        assert len({r.get('testName') for r in results})==count
        assert sum('OwnedPackedRuntimeIdentityTests.' in r.get('testName','') for r in results)==1
        expected='OwnedPackedUnavailableTests.' if mode=='scalar' else 'OwnedPackedWeightTests.'
        assert sum(expected in r.get('testName','') for r in results)==count-1
        suites.append(dict(mode=mode,passed=count,skipped=0,trx=pin(path),names=[r.get('testName') for r in results]))
    analysis=dict(passed=True,compiled_review=pin(BASE/'build-review.json'),product=built['product'],consumer=built['consumer'],
        suites=suites,resources=resources,release_admitted=False,no_model_execution=True,no_application_score=True,
        failed_release_controls=spec['failed_release_controls'],actual_87_weight_census_not_yet_observed=True,
        numerical_full_models_pending=True,application_comparison_pending=True)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),compiled_review=pin(BASE/'build-review.json'),
        collection=pin(folder/'capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),
        terminal_owners=receipt['identities'],reviewer=pin(Path(__file__)),
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),suites=[dict(mode=s['mode'],passed=s['passed']) for s in suites],product=built['product'],resources=resources)))


if __name__=='__main__':{'build':build,'capture':capture}[sys.argv[1]]()
