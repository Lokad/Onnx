"""Reuse the certified wall observer for the prospectively defined positional group."""
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
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-profile-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-slice-dense-conversion-profile-20260925'
PRIOR=ROOT/'artifacts/parakeet-managed-phase-amd-20260924'
OBSERVER=ROOT/'artifacts/parakeet-slice-materialization-profile-amd-20260924'
MODELS=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
APP=ROOT/'artifacts/parakeet-observed-dense-where-app-amd-20260924'
REMOTE_APP='/dev/shm/lokad-parakeet-observed-dense-where-app-20260924'
RESULTS=TOOLS.parent/'slice-dense-conversion-results'
GRAPH_REFERENCE=ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924/capture-collected/wall/graphs.json'
SOURCE=ROOT/'artifacts/parakeet-slice-dense-conversion-source-20260924'
MANIFEST='manifests/candidate-parakeet.json'
PRELUDE=transport.PRELUDE.replace(transport.REMOTE,REMOTE)
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def observer_scope():
    path=RESULTS/'observer-20260925.json'
    assert pin(path)['sha256']=='11dad08be4a439f63d9d5dfff4ff2169cb31ea9276d6284657cd1f25636a9bd0'
    scope=read(path);assert scope['passed'] and scope['no_rebuild'] and scope['no_inference']
    assert scope['all_original_bodies_flags_and_surface_exact'] and scope['original_data_methods']==697
    assert scope['current_model_closure']==pin(MODELS/'closed.json')
    assert scope['graph_metadata']==pin(GRAPH_REFERENCE) and scope['decoder_packed_bytes']==51461120
    for name,key in [('SampledAudio.dll','consumer'),('Lokad.Onnx.Data.dll','observed_data')]:
        assert pin(OBSERVER/'build-collected/runtime-observed'/name)==scope[key]
    return dict(passed=True,data=scope['observed_data'],consumer=scope['consumer'],no_rebuild=True,
        certified_review=pin(path),graph_metadata=scope['graph_metadata'],original_methods=697)


def prepare():
    assert not BASE.exists();scope=observer_scope()
    assert pin(MODELS/'closed.json')['sha256']=='5860fc366c3d976907f35f8d7a1832c10566fa08910af32c168dca7a63979842'
    model_proof=read(MODELS/'closed.json');assert model_proof['passed']
    assert model_proof['analysis']==pin(MODELS/'analysis.json')
    for name,wanted in model_proof['files'].items():assert pin(MODELS/name)==wanted,name
    identities=read(MODELS/'analysis.json')['identities']
    assert identities['selected']['Lokad.Onnx.dll']['sha256']=='f95a13c58354bf07f3b7926b72903c18b1a560a56673297cb9fe001d3541b592'
    assert identities['candidate']['Lokad.Onnx.dll']['sha256']=='49c3a958850d3e57daa2b7e29e6bd15ff9fc4f27af098d8ce44d8f20b9e065e8'
    assert pin(SOURCE/'prepared.json')['sha256']=='a9811c6369122eb8b53ab1d88d8329928a999d172d89786a1b393c3451e91e1e'
    original=read(SOURCE/'prepared.json')['before'];assert len(original)==427
    for name,wanted in original.items():assert pin(ROOT/name)==wanted,name
    assert pin(APP/'closed.json')['sha256']=='4ed28b9161ecdf4d31b2bffc0e7e25b1e539bf68a78b8f80e0edfb4255b86752'
    for name,wanted in read(APP/'closed.json')['files'].items():assert pin(APP/name)==wanted,name
    assert pin(RESULTS/'groups-20260925.json')['sha256']=='c7740ffb18f61eaf830dc6c99ff3f56e06bec2b945c8e95164f26b65f3bc3edc'
    definition=read(RESULTS/'groups-20260925.json');assert definition['passed'] and definition['no_inference']
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
    put('positional-groups.json',(RESULTS/'groups-20260925.json').read_bytes())
    put('expected-graphs.json',GRAPH_REFERENCE.read_bytes())
    put('prospective-plan.md',(ROOT/'.agent/m73-parakeet-slice-dense-conversion-20260924.md').read_bytes())
    manifest=read(APP/'collected'/MANIFEST);external={}
    for value in manifest['models'].values():external[value['path']]={k:value[k] for k in ['bytes','sha256']}
    for value in [manifest['reference'],*[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP+'/assets/'+value['path']]={k:value[k] for k in ['bytes','sha256']}
    for name in [MANIFEST,'runtime/protocol.py','runtime/campaign_processes.py']:
        external[REMOTE_APP+'/'+name]=pin(APP/'collected'/name)
    spec=dict(boot=1789634288.0,app=REMOTE_APP,manifest=MANIFEST,external=external,
        model_closure=pin(MODELS/'closed.json'),observer_closure=pin(OBSERVER/'closed.json'),
        phase_closure=pin(PRIOR/'closed.json'),groups=pin(RESULTS/'groups-20260925.json'),
        source_receipt=pin(SOURCE/'prepared.json'),source_files=original,
        core=runtime['core'],candidate_core=runtime['candidate_core'],data=scope['data'],consumer=scope['consumer'],
        require_complete_group_improves=True,require_all_24_kernels_improve=True,no_rebuild=True,
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    for path in TOOLS.glob('*.py'):ast.parse(path.read_text(encoding='utf8'),str(path))
    helpers=[TOOLS.parent/'managed-phase-amd'/n for n in ['run.py','remote.py','audit.py']]
    helpers += [RESULTS/'define_groups.py',TOOLS.parent/'ort-diagnosis-amd/run.py']
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},helpers={p.relative_to(ROOT).as_posix():pin(p) for p in helpers}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),no_rebuild=True,core=spec['core'],candidate_core=spec['candidate_core'])))


def prepared():
    value=read(BASE/'prepared.json')
    assert pin(BASE/'payload.tar.gz')==value['archive'] and pin(BASE/'bundle/spec.json')==value['spec']
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in value['helpers'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items():assert pin(BASE/'bundle'/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['source_files'].items():assert pin(ROOT/name)==wanted,name


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
