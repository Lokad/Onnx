"""Fix the Linux analyzer guard only, retaining the rejected first build."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
OLD_TOOLS=TOOLS.parent/'packed-logical-weight-probe'
OLD=ROOT/'artifacts/parakeet-packed-logical-weight-amd-20260925'
loader=importlib.util.spec_from_file_location('representation_original_run',OLD_TOOLS/'run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
pin,read,write,ssh,cases=original.pin,original.read,original.write,original.ssh,original.cases
BASE=ROOT/'artifacts/parakeet-packed-logical-weight-amd-v2-20260925'
REMOTE='/dev/shm/lokad-parakeet-packed-logical-weight-v2-20260925'
PRELUDE=original.PRELUDE.replace(original.REMOTE,REMOTE)
original.BASE,original.REMOTE,original.PRELUDE,original.TOOLS=BASE,REMOTE,PRELUDE,TOOLS
original.prior.BASE,original.prior.REMOTE,original.prior.PRELUDE=BASE,REMOTE,PRELUDE
original.transport.BASE,original.transport.REMOTE,original.transport.PRELUDE=BASE,REMOTE,PRELUDE


def initial():
    proof=read(TOOLS/'recovery.json');frozen=read(OLD/'prepared.json');folder=OLD/'build-collected'
    for key,path in {
        'original_preparation':OLD/'prepared.json','original_collection':folder/'build-collection.json',
        'original_transfer':OLD/'build-transfer.json','original_state':folder/'build-state.json',
        'original_source':OLD_TOOLS/'Program.cs.txt','corrected_source':TOOLS/'Program.cs.txt',
        'original_build_output':folder/'logs/consumer-build.stdout',
    }.items():assert pin(path)==proof[key],key
    for name,wanted in frozen['tools'].items():assert pin(OLD_TOOLS/name)==wanted,name
    for name,wanted in frozen['helpers'].items():assert pin(ROOT/name)==wanted,name
    before=(OLD_TOOLS/'Program.cs.txt').read_bytes();after=(TOOLS/'Program.cs.txt').read_bytes()
    assert before.count(proof['before'].encode())==1 and after.count(proof['after'].encode())==1
    assert after.replace(proof['after'].encode(),proof['before'].encode())==before
    for name in ['PackedLogicalWeight.cs.txt','vm.py']:assert (TOOLS/name).read_bytes()==(OLD_TOOLS/name).read_bytes()
    receipt=read(folder/'build-collection.json');transfer=read(OLD/'build-transfer.json');state=read(folder/'build-state.json')
    assert transfer['passed'] and transfer['archive']==pin(OLD/'build-results.tar.gz')
    assert transfer['collection']==pin(folder/'build-collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['state']==pin(folder/'build-state.json')
    assert state['complete'] and state['code']==0
    assert [(r['name'],r['code']) for r in state['runs']]==[('sdk-version',0),('consumer-restore',0),('consumer-build',0)]
    output=(folder/'logs/consumer-build.stdout').read_text()
    assert 'warning CA1416' in output and '1 Warning(s)' in output and '0 Error(s)' in output
    assert not (OLD/'build-review.json').exists() and not (OLD/'capture-deployment.json').exists()
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    return proof


def prepared():initial();original.prepared()


if __name__=='__main__':
    action=sys.argv[1];initial()
    if action=='prepare':original.prepare()
    else:
        prepared()
        if action=='stage':original.transport.stage()
        else:
            kind=sys.argv[2];assert kind in ['build','capture']
            if action=='launch':
                if kind=='capture':assert read(BASE/'build-review-transferred.json')['passed']
                original.transport.launch(kind)
            else:{'observe':original.prior.observe,'collect':original.prior.collect}[action](kind)
