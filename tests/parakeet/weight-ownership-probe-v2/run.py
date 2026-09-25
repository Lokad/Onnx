"""Correct one consumer compile error in a new namespace; retain the failed build."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
OLD_TOOLS=TOOLS.parent/'weight-ownership-probe'
OLD=ROOT/'artifacts/parakeet-weight-ownership-amd-20260925'
loader=importlib.util.spec_from_file_location('ownership_original_run',OLD_TOOLS/'run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
pin,read,write,ssh=original.pin,original.read,original.write,original.ssh
BASE=ROOT/'artifacts/parakeet-weight-ownership-amd-v2-20260925'
REMOTE='/dev/shm/lokad-parakeet-weight-ownership-v2-20260925'
PRELUDE=original.PRELUDE.replace(original.REMOTE,REMOTE)
original.BASE,original.REMOTE,original.PRELUDE,original.TOOLS=BASE,REMOTE,PRELUDE,TOOLS
original.transport.BASE,original.transport.REMOTE,original.transport.PRELUDE=BASE,REMOTE,PRELUDE
references=original.references


def initial():
    proof=read(TOOLS/'recovery.json');manifest=read(OLD/'prepared.json');folder=OLD/'build-collected'
    assert proof['original_preparation']==pin(OLD/'prepared.json')
    assert proof['original_collection']==pin(folder/'build-collection.json')
    assert proof['original_transfer']==pin(OLD/'build-transfer.json')
    assert proof['original_state']==pin(folder/'build-state.json')
    assert proof['original_source']==pin(OLD_TOOLS/'Program.cs.txt')
    assert proof['corrected_source']==pin(TOOLS/'Program.cs.txt')
    before=(OLD_TOOLS/'Program.cs.txt').read_text();after=(TOOLS/'Program.cs.txt').read_text()
    assert before.count(proof['before'])==1 and after.count(proof['after'])==1
    assert after.replace(proof['after'],proof['before'])==before
    assert (TOOLS/'vm.py').read_bytes()==(OLD_TOOLS/'vm.py').read_bytes()
    for name,wanted in manifest['tools'].items():assert pin(OLD_TOOLS/name)==wanted,name
    for name,wanted in manifest['helpers'].items():assert pin(ROOT/name)==wanted,name
    receipt=read(folder/'build-collection.json');transfer=read(OLD/'build-transfer.json');state=read(folder/'build-state.json')
    assert transfer['passed'] and transfer['archive']==pin(OLD/'build-results.tar.gz')
    assert transfer['collection']==pin(folder/'build-collection.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['state']==pin(folder/'build-state.json')
    assert state['complete'] and state['code']==1
    assert [(r['name'],r['code']) for r in state['runs']]==[('sdk-version',0),('consumer-restore',0),('consumer-build',1)]
    assert 'error CS0411' in (folder/'logs/consumer-build.stdout').read_text()
    assert not (folder/'built.json').exists() and not (OLD/'capture-deployment.json').exists()
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    return proof


def prepared():
    initial();original.prepared()


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
            else:{'observe':original.observe,'collect':original.collect}[action](kind)
