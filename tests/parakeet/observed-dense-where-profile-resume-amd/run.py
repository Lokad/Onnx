"""Complete only the unstarted candidate after a retained memory refusal."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
ORIGINAL_TOOLS=TOOLS.parent/'observed-dense-where-profile-amd'
loader=importlib.util.spec_from_file_location('initial_profile',ORIGINAL_TOOLS/'run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
pin,read,write,ssh=original.pin,original.read,original.write,original.ssh
ORIGINAL=original.BASE
BASE=ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-observed-dense-where-profile-resume-20260924'
APP,PRIOR,MATCHED,MANIFEST=original.APP,original.PRIOR,original.MATCHED,original.MANIFEST
observer_scope=original.observer_scope
transport=original.transport
PRELUDE=original.PRELUDE.replace(original.REMOTE,REMOTE)
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def initial():
    original.prepared()
    folder=ORIGINAL/'capture-collected';receipt=read(folder/'capture-collection.json')
    transfer=read(ORIGINAL/'capture-transfer.json');state=read(folder/'capture-state.json')
    assert transfer['passed'] and transfer['archive']==pin(ORIGINAL/'capture-results.tar.gz')
    assert transfer['collection']==pin(folder/'capture-collection.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['state']==pin(folder/'capture-state.json')
    assert state['complete'] and state['code']==1 and state['supervisor']==read(ORIGINAL/'capture-deployment.json')
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert [r['name'] for r in state['runs']]==['control','wall']
    control,missing=state['runs'];limits=read(ORIGINAL/'bundle/spec.json')['capture_limits']
    assert control['complete'] and control['code']==0 and control['samples']>0
    assert not missing['complete'] and missing['code'] is None and missing['samples']==0 and not missing['members']
    assert not any(k in missing for k in ['owner','ready','seconds'])
    assert missing['preflight']['available']<limits['available_before']
    assert missing['preflight']['tmpfs']>=limits['tmpfs_before']
    assert 'assert preflight[' in state['error'] and state['error'].endswith('AssertionError\n')
    assert not any(p.name.startswith('wall.') for p in (folder/'logs').iterdir()) and not (folder/'wall').exists()
    return dict(collection=pin(folder/'capture-collection.json'),transfer=pin(ORIGINAL/'capture-transfer.json'),
        archive=pin(ORIGINAL/'capture-results.tar.gz'),state=pin(folder/'capture-state.json'),
        spec=pin(ORIGINAL/'bundle/spec.json'),deployment=pin(ORIGINAL/'capture-deployment.json'),
        candidate_never_started=True,original_code=1)


def prepare():
    assert not BASE.exists();first=initial()
    remote=(ORIGINAL_TOOLS/'remote.py').read_text(encoding='utf8')
    assert remote.count("for name in ['control','wall']:")==1
    assert (TOOLS/'remote.py').read_text(encoding='utf8')==remote.replace("for name in ['control','wall']:","for name in ['wall']:")
    BASE.mkdir();bundle=BASE/'bundle';shutil.copytree(ORIGINAL/'bundle',bundle)
    (bundle/'spec.json').unlink()
    shutil.copy2(TOOLS/'remote.py',bundle/'remote.py')
    shutil.copy2(TOOLS/'README.md',bundle/'recovery.md')
    shutil.copy2(ROOT/'.agent/m70-parakeet-observed-dense-where-20260924.md',bundle/'recovery-plan.md')
    write(bundle/'initial.json',first)
    spec=read(ORIGINAL/'bundle/spec.json');spec['initial']=first
    for name in ['capture-collection.json','capture-state.json','spec.json']:
        spec['external'][original.REMOTE+'/'+name]=pin(ORIGINAL/'capture-collected'/name)
    spec['files']={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    for path in TOOLS.glob('*.py'):ast.parse(path.read_text(encoding='utf8'),str(path))
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),initial=first,
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        initial_preparation=pin(ORIGINAL/'prepared.json')))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),candidate_only=True)))


def prepared():
    first=initial();value=read(BASE/'prepared.json')
    assert value['initial']==first and value['initial_preparation']==pin(ORIGINAL/'prepared.json')
    assert pin(BASE/'payload.tar.gz')==value['archive'] and pin(BASE/'bundle/spec.json')==value['spec']
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    spec=read(BASE/'bundle/spec.json');before=read(ORIGINAL/'bundle/spec.json')
    assert spec['initial']==first
    assert {k:v for k,v in spec.items() if k not in ['files','external','initial']}=={k:v for k,v in before.items() if k not in ['files','external']}
    assert all(spec['external'][k]==v for k,v in before['external'].items())
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


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
    else:
        prepared()
        if action=='stage':transport.stage()
        elif action=='observe':observe()
        else:dict(launch=transport.launch,collect=transport.collect)[action]('capture')
