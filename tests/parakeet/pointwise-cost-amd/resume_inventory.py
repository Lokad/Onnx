"""Finish only the missing four-argument inventory; preserve the failed build."""
import base64
import json
from pathlib import Path
import sys
from run import BASE, TOOLS, PRELUDE, pin, read, write, ssh, prepared, prior

REMOTE_SCRIPT = r'''
import os,time,traceback
from pathlib import Path
import common
BASE=Path(__file__).resolve().parent;common.BASE=BASE
own=common.psutil.Process();own.cpu_affinity([0]);common.idle()
spec=common.verify();old=common.read(BASE/'build-state.json')
assert old['complete'] and old['code']==1 and not common.live(old['supervisor'])
assert all(r['complete'] and r['code']==0 for r in old['runs'][:-1])
assert old['runs'][-1]['name']=='inventory' and old['runs'][-1]['code']==-6
assert not (BASE/'inventory-state.json').exists() and not (BASE/'built.json').exists()
state=dict(kind='inventory',complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[],started=time.time(),
 failed_state=common.pin(BASE/'build-state.json'),script=common.pin(Path(__file__)),
 inspector=common.pin(BASE/'bridge-source/bin/Release/net10.0/Bridge.dll'),
 observed_core=common.pin(BASE/'runtime/Lokad.Onnx.dll'))
common.save(BASE/'inventory-state.json',state)
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_','parakeet_'))}
env.pop('PYTHONOPTIMIZE',None)
try:
 common.job(state,'inventory-recovered',[common.DOTNET,BASE/'bridge-source/bin/Release/net10.0/Bridge.dll',
  spec['prior'],BASE/'runtime',BASE/'logs/instructions.json',spec['prior']],env,BASE,spec['build_limits'],spec)
 files={p.name:common.pin(p) for p in (BASE/'runtime').iterdir() if p.is_file()}
 assert files['Lokad.Onnx.dll']==state['observed_core']
 common.save(BASE/'built.json',dict(passed=True,runtime=files,product={n:files[n] for n in spec['before_product']},
  consumer=files['SampledAudio.dll'],inventory=common.pin(BASE/'logs/instructions.json')))
 common.verify();state['code']=0
except BaseException:
 state.update(code=1,error=traceback.format_exc());traceback.print_exc()
finally:
 state.update(complete=True,ended=time.time());common.save(BASE/'inventory-state.json',state)
raise SystemExit(state['code'])
'''


def launch():
    prepared()
    compile(REMOTE_SCRIPT,'inventory-resume','exec')
    assert not (BASE/'inventory-deployment.json').exists()
    write(BASE/'inventory-resume.json',dict(script=REMOTE_SCRIPT,tool=pin(Path(__file__)),
        failed_collection=pin(BASE/'build-collected/build-collection.json'),
        reason='The compiled inspector expects four arguments; the reused worker supplied three. No inference ran.'))
    encoded = base64.b64encode(REMOTE_SCRIPT.encode()).decode()
    value = ssh(PRELUDE+f'''
from remote import verify,idle,live,read
import base64
verify();idle();state=read(base/'build-state.json')
assert state['complete'] and state['code']==1 and not live(state['supervisor'])
with (base/'inventory-resume.py').open('xb') as f:f.write(base64.b64decode({encoded!r}))
env=dict(os.environ,PYTHONPATH={prior.transport.transport.SITE!r},PYTHONDONTWRITEBYTECODE='1')
env.pop('PYTHONOPTIMIZE',None)
with (base/'inventory-supervisor.stdout').open('x') as out,(base/'inventory-supervisor.stderr').open('x') as err:
 p=subprocess.Popen([sys.executable,'-B',str(base/'inventory-resume.py')],cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time())
print(json.dumps(value))
''')
    write(BASE/'inventory-deployment.json',value)
    print(json.dumps(value))


def audit():
    prepared()
    assert not (BASE/'build-review.json').exists()
    resume = read(BASE/'inventory-resume.json')
    assert resume['tool'] == pin(Path(__file__)) and resume['script'] == REMOTE_SCRIPT
    spec = read(BASE/'bundle/spec.json')
    folders = {k:BASE/(k+'-collected') for k in ['build','inventory']}
    states, resources = {}, []
    for kind, folder in folders.items():
        transfer = read(BASE/(kind+'-transfer.json'))
        receipt = read(folder/(kind+'-collection.json'))
        assert transfer['passed'] and transfer['archive'] == pin(BASE/(kind+'-results.tar.gz'))
        assert transfer['collection'] == pin(folder/(kind+'-collection.json')) and receipt['terminal']
        for name,wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
        state = read(folder/(kind+'-state.json')); states[kind] = state
        assert state['complete'] and state['supervisor'] == read(BASE/(kind+'-deployment.json'))
        assert receipt['state'] == pin(folder/(kind+'-state.json')) and receipt['code'] == state['code']
        assert receipt['identities'] == [state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
        assert pin(folder/'spec.json') == pin(BASE/'bundle/spec.json')
        for name,wanted in spec['files'].items(): assert pin(folder/name) == wanted, name
        for job in state['runs']:
            limits=spec['build_limits']
            assert job['complete'] and job['seconds']<limits['seconds']
            assert job['preflight']['available']>=limits['available_before'] and job['preflight']['tmpfs']>=limits['tmpfs_before']
            samples=[json.loads(s) for s in (folder/'logs'/(job['name']+'.resources.jsonl')).read_text().splitlines()]
            assert len(samples)==job['samples']>0
            for sample in samples:
                assert sample['seconds']<limits['seconds'] and sample['rss']<limits['rss']
                assert min(sample['available'],sample['tmpfs'])>=spec['minimum_free'] and sample['output']<spec['output_limit']
                assert sample['rss']==sum(m['rss'] for m in sample['members'])
                for member in sample['members']:
                    assert job['members'][str(member['pid'])]==member['birth'] and member['affinity']==[2]
                    assert member['threads'] and all(t==[2] for t in member['threads'])
            resources.append(dict(name=job['name'],samples=len(samples),seconds=job['seconds'],peak_rss=max(s['rss'] for s in samples)))
    old, recovered = states['build'], states['inventory']
    assert old['code']==1 and [r['code'] for r in old['runs']]==[0,0,0,0,0,-6]
    assert [r['name'] for r in old['runs']]==['sdk-version','core-restore','core-build','bridge-restore','bridge-build','inventory']
    assert 'System.ArgumentException: old bin, corrected bin, new result' in (folders['build']/'logs/inventory.stderr').read_text()
    assert recovered['code']==0 and len(recovered['runs'])==1 and recovered['runs'][0]['code']==0
    assert recovered['runs'][0]['name']=='inventory-recovered' and old['ended']<recovered['started']
    assert recovered['failed_state']==pin(folders['build']/'build-state.json')
    folder=folders['inventory'];built=read(folder/'built.json')
    assert built['passed'] and built['inventory']==pin(folder/'logs/instructions.json')
    assert built['runtime'].keys()==spec['runtime'].keys()
    for name,wanted in built['runtime'].items():
        assert pin(folder/'runtime'/name)==pin(folders['build']/'runtime'/name)==wanted
    assert [n for n in built['runtime'] if built['runtime'][n]!=spec['runtime'][n]]==['Lokad.Onnx.dll']
    assert built['product']=={n:built['runtime'][n] for n in spec['before_product']} and built['consumer']==spec['consumer']
    assert recovered['observed_core']==built['product']['Lokad.Onnx.dll']
    inventory=read(folder/'logs/instructions.json');assert inventory['inventory_complete']
    assert inventory['release']['sha256']==spec['before_product']['Lokad.Onnx.dll']['sha256']
    assert [r['assembly'] for r in inventory['observations']]==['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']
    scope=[]
    for row in inventory['observations']:
        core=row['assembly']=='Lokad.Onnx.dll'
        assert row['before_sha256']==spec['before_product'][row['assembly']]['sha256']
        assert row['after_sha256']==built['product'][row['assembly']]['sha256']
        assert row['methods']==(3286 if core else 697) and not row['removed']
        assert row['unchanged_methods']==row['methods']-(3 if core else 0)
        if core:
            assert len(row['differences'])==3 and {n.split('::')[1] for n in row['differences']}==set(spec['changed_methods'])
            assert all(n.startswith('Lokad.Onnx.Tensor`1[T]::') for n in row['differences'])
            assert row['added'] and all(n.startswith(('Lokad.Onnx.PointwiseCostProbe::','Lokad.Onnx.PointwiseCostProbe+')) for n in row['added'])
        else: assert not row['differences'] and not row['added']
        assert row['public_surface_equal'] and row['public_surface']==row['public_surface_after']
        assert row['assembly_attributes_before']==row['assembly_attributes_after']
        assert all(row['method_flags_after'][n]==v for n,v in row['method_flags_before'].items())
        assert set(row['candidate_methods'])==set(row['differences']+row['added'])
        scope.append(dict(assembly=row['assembly'],unchanged=row['unchanged_methods'],changed=row['differences'],added=row['added']))
    warnings=[]
    for job in [*old['runs'][:-1],*recovered['runs']]:
        text=(folder/'logs'/(job['name']+'.stdout')).read_text()+(folder/'logs'/(job['name']+'.stderr')).read_text()
        assert ': error ' not in text
        warnings.extend(s for s in text.splitlines() if ': warning ' in s)
    assert len(warnings)==4 and all('Zzz.WideProjectionEntry.cs(20,' in s and 'warning CS8604:' in s for s in warnings)
    result=dict(passed=True,built=pin(folder/'built.json'),scope=scope,resources=resources,warnings=warnings,
        source_reversal_checked=True,consumer_unchanged=True,data_unchanged=True,diagnostic_only=True,release_admitted=False,
        failed_build=pin(folders['build']/'build-collection.json'),recovery=pin(folder/'inventory-collection.json'),
        reason=resume['reason'],reviewer=pin(Path(__file__)))
    write(BASE/'build-review.json',result)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    transferred=ssh(PRELUDE+f'''
from remote import verify,read,live,pin
import base64
verify();state=read(base/'inventory-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={result['built']!r}
with (base/'build-review.json').open('xb') as f:f.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert transferred['review']==pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',transferred)
    print(json.dumps(dict(**transferred,scope=scope)))


if __name__ == '__main__':
    action=sys.argv[1]
    if action=='launch':launch()
    elif action=='audit':audit()
    else:
        prepared()
        {'observe':prior.observe,'collect':prior.collect}[action]('inventory')
