"""Resume only the unstarted census mode after the original memory refusal."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('original_counter_transport',TOOLS.parent/'packed-final-row-census/run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
pin,read,write,ssh=original.pin,original.read,original.write,original.ssh
FIRST=original.BASE;FIRST_REMOTE=original.REMOTE
BASE=ROOT/'artifacts/parakeet-packed-final-row-census-resume-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-packed-final-row-census-resume-20260925'
PRELUDE=original.PRELUDE.replace(FIRST_REMOTE,REMOTE)
prior=original.prior;prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport=original.transport;transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE
JOBS=['census-256'];ALL_JOBS=['census-512',*JOBS]


def references():
    original.prepared()
    receipt=read(FIRST/'capture-collected/capture-collection.json');state=read(FIRST/'capture-collected/capture-state.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['state']==pin(FIRST/'capture-collected/capture-state.json')
    transfer=read(FIRST/'capture-transfer.json')
    assert transfer['passed'] and transfer['archive']==pin(FIRST/'capture-results.tar.gz') and transfer['collection']==pin(FIRST/'capture-collected/capture-collection.json')
    for name,wanted in receipt['files'].items():assert pin(FIRST/'capture-collected'/name)==wanted,name
    assert state['complete'] and state['code']==1 and state['supervisor']==read(FIRST/'capture-deployment.json')
    assert [r['name'] for r in state['runs']]==['census-512','census-256']
    done,missing=state['runs'];assert done['complete'] and done['code']==0 and done['samples']>0
    assert not missing['complete'] and missing['code'] is None and missing['samples']==0 and not missing['members'] and 'owner' not in missing
    spec=read(FIRST/'bundle/spec.json');assert missing['preflight']['available']<spec['capture_limits']['available_before']
    assert missing['preflight']['tmpfs']>=spec['capture_limits']['tmpfs_before'] and 'preflight' in state['error']
    result=read(FIRST/'capture-collected/probe/512/result.json');assert result['passed'] and result['mode']=='512'
    assert read(FIRST/'build-review.json')['passed'] and pin(FIRST/'build-review.json')['sha256']=='57d9cfd8eb9844167453b2e4cff68239cbd2819da04255904b3eb76cbab14f98'
    return spec,receipt,state


def prepare():
    assert not BASE.exists();old,receipt,state=references();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,source):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as f:f.write(source.read_bytes())
    for name in old['files']:
        if name in ['remote.py','README.md','plan.md']:continue
        put(name,FIRST/'bundle'/name)
    put('remote.py',TOOLS/'vm.py');put('README.md',TOOLS/'README.md');put('plan.md',ROOT/'.agent/m78-parakeet-packed-final-row-20260925.md')
    for source,name in [('capture-collected/capture-collection.json','collection.json'),('capture-collected/capture-state.json','state.json'),
                        ('bundle/spec.json','spec.json'),('capture-transfer.json','transfer.json'),('build-review.json','build-review.json')]:
        put('evidence/initial/'+name,FIRST/source)
    for p in (FIRST/'capture-collected/probe/512').iterdir():put('probe/512/'+p.name,p)
    for suffix in ['stdout','stderr','resources.jsonl']:put('evidence/initial/census-512.'+suffix,FIRST/'capture-collected/logs'/('census-512.'+suffix))
    initial=dict(collection=pin(FIRST/'capture-collected/capture-collection.json'),state=pin(FIRST/'capture-collected/capture-state.json'),
        archive=pin(FIRST/'capture-results.tar.gz'),transfer=pin(FIRST/'capture-transfer.json'),spec=pin(FIRST/'bundle/spec.json'),
        build_review=pin(FIRST/'build-review.json'),original_code=1,completed_job='census-512',remaining_jobs=JOBS,
        supervisor=state['supervisor'],selected_result=pin(FIRST/'capture-collected/probe/512/result.json'))
    built=read(FIRST/'build-collected/built.json');external=dict(old['external'])
    external.update({FIRST_REMOTE+'/runtime/'+name:value for name,value in built['runtime'].items()})
    for name in ['capture-collection.json','capture-state.json','build-review.json']:external[FIRST_REMOTE+'/'+name]=pin(FIRST/('build-review.json' if name=='build-review.json' else 'capture-collected/'+name))
    spec=dict(old,initial=initial,jobs=JOBS,all_jobs=ALL_JOBS,external=external,original_runtime=FIRST_REMOTE+'/runtime',
        binding_runtime=built['runtime'],initial_remote=FIRST_REMOTE,preflight_wait_seconds=900,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},original_prepared=pin(FIRST/'prepared.json'),initial=initial))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),retained='census-512',remaining=JOBS)))


def prepared():
    references();value=read(BASE/'prepared.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json') and value['original_prepared']==pin(FIRST/'prepared.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in read(BASE/'bundle/spec.json')['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    else:
        prepared()
        if action=='stage':transport.stage()
        else:
            kind=sys.argv[2];assert kind in ['build','capture']
            if action=='launch':
                if kind=='capture':assert read(BASE/'build-review-transferred.json')['passed']
                transport.launch(kind)
            else:{'observe':prior.observe,'collect':prior.collect}[action](kind)
