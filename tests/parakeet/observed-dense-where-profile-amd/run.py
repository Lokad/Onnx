"""Reuse the exact wall observer to test one observed masking mechanism."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('phase_transport',TOOLS.parent/'managed-phase-amd/run.py')
transport=importlib.util.module_from_spec(loader);loader.loader.exec_module(transport)
pin,read,write,ssh=transport.pin,transport.read,transport.write,transport.ssh
BASE=ROOT/'artifacts/parakeet-observed-dense-where-profile-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-observed-dense-where-profile-20260924'
PRIOR=ROOT/'artifacts/parakeet-managed-phase-amd-20260924'
OBSERVER=ROOT/'artifacts/parakeet-slice-materialization-profile-amd-20260924'
MODELS=ROOT/'artifacts/parakeet-observed-dense-where-models-amd-20260924'
NUMERICS=ROOT/'artifacts/parakeet-observed-dense-where-numerics-amd-20260924'
BUILD=ROOT/'artifacts/parakeet-observed-dense-where-inventory-amd-20260924'
RELEASE=ROOT/'artifacts/parakeet-validated-composition-root-amd-20260924'
APP=ROOT/'artifacts/parakeet-validated-composition-app-amd-20260924'
REMOTE_APP='/dev/shm/lokad-parakeet-validated-composition-app-20260924'
MATCHED=ROOT/'artifacts/parakeet-masking-padding-attribution-20260924'
MANIFEST='manifests/candidate-parakeet.json'
PRELUDE=transport.PRELUDE.replace(transport.REMOTE,REMOTE)
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def profile_proof(folder,digest):
    assert pin(folder/'closed.json')['sha256']==digest
    proof=read(folder/'closed.json');assert proof['passed']
    for key,name in [('analysis','analysis.json'),('build_review','build-review.json'),
        ('collection','capture-collected/capture-collection.json'),('transfer','capture-transfer.json')]:
        assert proof[key]==pin(folder/name),name
    for kind in ['build','capture']:
        for name,wanted in read(folder/f'{kind}-collected/{kind}-collection.json')['files'].items():
            assert pin(folder/f'{kind}-collected'/name)==wanted,name


def observer_scope():
    profile_proof(PRIOR,'2a9d9342acfffede564e1a6870a8027322049e1b5087c005c260d9a054e0c2a3')
    profile_proof(OBSERVER,'69693be5e42f0dfa3bd66ea36e42c714e63a1d50a020dd9ec04aa81fbda9b48d')
    for folder,digest in [(RELEASE,'c7a1d2e11566e6eeeb965de6c9cedbf47df479fd51f194c797af412446281609'),
        (BUILD,'7d5296b266feeacbf9c52bf57a689de2cdf0a839045130f15912912f5da1b7bd'),
        (NUMERICS,'0dfb5dcb4d5c5793c7c175837bde6f9469e791f62c02cf1b4df679ba4d81f0db')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    old_path=PRIOR/'build-collected/inventory/instructions.json'
    current_path=RELEASE/'collected/inventory/instructions.json'
    old=next(r for r in read(old_path)['observations'] if r['assembly']=='Lokad.Onnx.Data.dll')
    current=next(r for r in read(current_path)['observations'] if r['assembly']=='Lokad.Onnx.Data.dll')
    assert len(old['normalized_methods'])==len(current['normalized_methods'])==697
    assert old['normalized_methods']==current['normalized_methods']
    assert old['method_flags_before']==current['method_flags_before']
    assert current['before_sha256']==read(BUILD/'analysis.json')['measured']['Lokad.Onnx.Data.dll']['sha256']
    for name,wanted in read(PRIOR/'bundle/spec.json')['data_source'].items():
        assert pin(ROOT/'src/Lokad.Onnx.Data'/name)==wanted,name
    row,=read(OBSERVER/'build-collected/inventory/instructions.json')['observations']
    assert row['assembly']=='SampledAudio.dll' and row['methods']==164 and row['unchanged_methods']==163
    assert row['method_flags_before']==row['method_flags_after'] and row['public_surface_equal']
    assert not row['added'] and not row['removed'] and len(row['differences'])==1
    runtime=OBSERVER/'build-collected/runtime-observed'
    assert pin(runtime/'SampledAudio.dll')['sha256']=='38ab5c7e65aa00ace50e8704348830286047e0c96ebc0a04b66c6e54d063899c'
    assert pin(runtime/'Lokad.Onnx.Data.dll')['sha256']=='a2a0b4901ba8e4270b57e2b176d7a656e739886275ec66d0ca036807eea16e3e'
    return dict(passed=True,current_data_methods=697,current_data_bodies_and_flags_exact=True,
        original_data_sources_exact=True,original_inventory=pin(old_path),current_inventory=pin(current_path),
        original_observer_review=pin(PRIOR/'build-review.json'),consumer_review=pin(OBSERVER/'build-review.json'),
        consumer=pin(runtime/'SampledAudio.dll'),data=pin(runtime/'Lokad.Onnx.Data.dll'),no_rebuild=True)


def prepare():
    assert not BASE.exists();scope=observer_scope()
    assert pin(MODELS/'closed.json')['sha256']=='ff9a7a2b5f1156bd365d0f51564f1160e72a82f632ce6e544bb7f86a5663e337'
    model_proof=read(MODELS/'closed.json');assert model_proof['passed']
    assert model_proof['analysis']==pin(MODELS/'analysis.json')
    for name,wanted in model_proof['files'].items():assert pin(MODELS/name)==wanted,name
    identities=read(MODELS/'analysis.json')['identities']
    numerical=read(NUMERICS/'analysis.json')['products']
    assert identities==dict(selected=numerical['current'],candidate=numerical['candidate'])
    assert pin(APP/'closed.json')['sha256']=='f04c09fbc6c0455c4420d6680d60bb4f8fc5cac9dda2f2507768ba94b86335e4'
    for name,wanted in read(APP/'closed.json')['files'].items():assert pin(APP/name)==wanted,name
    assert pin(MATCHED/'closed.json')['sha256']=='0299d2e1d273b2651d45209cb7c9dffb623ae4149cf0cb5eed23eaf89d842e76'
    assert read(MATCHED/'closed.json')['analysis']==pin(MATCHED/'analysis.json')
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data if isinstance(data,bytes) else data.encode())
    for name,role in [('runtime-control','selected'),('runtime-observed','candidate')]:
        for path in (OBSERVER/'build-collected/runtime-observed').iterdir():
            if not path.is_file():continue
            source=MODELS/'collected/runtimes'/role/path.name if path.name=='Lokad.Onnx.dll' else path
            put(name+'/'+path.name,source.read_bytes())
        assert pin(bundle/name/'Lokad.Onnx.dll')==identities[role]['Lokad.Onnx.dll']
        assert pin(bundle/name/'Lokad.Onnx.Data.dll')==scope['data']
    runtime=dict(core=identities['selected']['Lokad.Onnx.dll'],candidate_core=identities['candidate']['Lokad.Onnx.dll'],
        data=scope['data'],consumer=scope['consumer'],no_rebuild=True,
        runtime_files={p.relative_to(bundle).as_posix():pin(p) for name in ['runtime-control','runtime-observed'] for p in (bundle/name).iterdir()})
    write(bundle/'runtime.json',runtime)
    write(bundle/'observer-review.json',dict(**scope,runtime=pin(bundle/'runtime.json'),model_closure=pin(MODELS/'closed.json')))
    for name in ['remote.py','README.md']:put(name,(TOOLS/name).read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('matched-input-groups.json',(MATCHED/'analysis.json').read_bytes())
    put('prospective-plan.md',(ROOT/'.agent/m70-parakeet-observed-dense-where-20260924.md').read_bytes())
    manifest=read(APP/'collected'/MANIFEST);external={}
    for value in manifest['models'].values():external[value['path']]={k:value[k] for k in ['bytes','sha256']}
    for value in [manifest['reference'],*[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP+'/assets/'+value['path']]={k:value[k] for k in ['bytes','sha256']}
    for name in [MANIFEST,'runtime/protocol.py','runtime/campaign_processes.py']:
        external[REMOTE_APP+'/'+name]=pin(APP/'collected'/name)
    spec=dict(boot=1789634288.0,app=REMOTE_APP,manifest=MANIFEST,external=external,
        model_closure=pin(MODELS/'closed.json'),observer_closure=pin(OBSERVER/'closed.json'),
        phase_closure=pin(PRIOR/'closed.json'),matched_closure=pin(MATCHED/'closed.json'),
        core=runtime['core'],candidate_core=runtime['candidate_core'],data=scope['data'],consumer=scope['consumer'],
        minimum_where_gain=.80,require_each_family_improves=True,no_rebuild=True,
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    for path in TOOLS.glob('*.py'):ast.parse(path.read_text(encoding='utf8'),str(path))
    helpers=[TOOLS.parent/'managed-phase-amd'/n for n in ['run.py','remote.py','audit.py','compare_masking_padding.py','compare_slices.py']]
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},helpers={p.relative_to(ROOT).as_posix():pin(p) for p in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),no_rebuild=True,core=spec['core'],candidate_core=spec['candidate_core'])))


def prepared():
    value=read(BASE/'prepared.json')
    assert pin(BASE/'payload.tar.gz')==value['archive'] and pin(BASE/'bundle/spec.json')==value['spec']
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


def observe():
    assert not (BASE/'closed.json').exists()
    result=ssh(PRELUDE+f'''
from remote import read,live
state=read(base/'capture-state.json') if (base/'capture-state.json').exists() else None
ids=[{read(BASE/'capture-deployment.json')!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},error=state and state.get('error'))))
''')
    with (BASE/'capture-observations.jsonl').open('a') as stream:stream.write(json.dumps(result)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='inspect':print(json.dumps(observer_scope()))
    else:
        prepared()
        if action=='stage':transport.stage()
        elif action=='observe':observe()
        else:dict(launch=transport.launch,collect=transport.collect)[action]('capture')
