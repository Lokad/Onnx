"""Diagnose the failed actual-encoder eligibility prediction without inference."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('census_transport',TOOLS.parent/'owned-packed-weight-census/run.py')
prior=importlib.util.module_from_spec(loader);loader.loader.exec_module(prior)
ORIGINAL=prior.BASE
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-selection-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-owned-packed-weight-selection-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
original_references=prior.references


def references():
    value=original_references()
    receipt=ORIGINAL/'capture-collected/capture-collection.json'
    assert pin(receipt)['sha256']=='6580a563bd6937b1868941a564121ce36d7f1218e47d146afd946c06c0c48f41'
    failed=read(receipt);assert failed['terminal'] and failed['code']==1
    for name,wanted in failed['files'].items():assert pin(receipt.parent/name)==wanted,name
    state=read(receipt.parent/'capture-state.json')
    assert state['complete'] and len(state['runs'])==1 and state['runs'][0]['code']==-6
    assert '87 equal-sized replacement payloads' in (receipt.parent/'logs/census-512.stderr').read_text()
    return value


prior.references=references
prior.BASE,prior.REMOTE,prior.PRELUDE,prior.TOOLS=BASE,REMOTE,PRELUDE,TOOLS
for module in [prior.prior,prior.transport]:module.BASE,module.REMOTE,module.PRELUDE=BASE,REMOTE,PRELUDE
prepare,prepared=prior.prepare,prior.prepared


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    else:
        prepared()
        if action=='stage':prior.transport.stage()
        else:
            kind=sys.argv[2];assert kind in ['build','capture']
            if action=='launch':
                if kind=='capture':assert read(BASE/'build-review-transferred.json')['passed']
                prior.transport.launch(kind)
            else:{'observe':prior.prior.observe,'collect':prior.prior.collect}[action](kind)
