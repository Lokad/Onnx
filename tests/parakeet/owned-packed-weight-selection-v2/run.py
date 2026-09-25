"""Fix the diagnostic tensor metadata access; retain its failed first build."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('selection_original',TOOLS.parent/'owned-packed-weight-selection/run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
FAILED=original.BASE;prior=original.prior
BASE=ROOT/'artifacts/parakeet-owned-packed-weight-selection-v2-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-owned-packed-weight-selection-v2-20260925'
PRELUDE=prior.PRELUDE.replace(prior.REMOTE,REMOTE)
pin,read,write,ssh=prior.pin,prior.read,prior.write,prior.ssh
first_references=prior.references


def references():
    value=first_references();receipt=read(FAILED/'build-collected/build-collection.json')
    assert receipt['terminal'] and receipt['code']==1
    for name,wanted in receipt['files'].items():assert pin(FAILED/'build-collected'/name)==wanted,name
    state=read(FAILED/'build-collected/build-state.json')
    assert state['complete'] and state['code']==1 and len(state['runs'])==3 and state['runs'][-1]['code']==1
    output=(FAILED/'build-collected/logs/consumer-build.stdout').read_text()
    assert output.count('error CS1061:')==2 and "definition for 'Dims'" in output
    before='kind = tensor.GetType().FullName, dimensions = tensor.Dims, layout, consumers,'
    after='kind = tensor.GetType().ToString(), dimensions = tensor.Dimensions.ToArray(), layout, consumers,'
    source=(TOOLS.parent/'owned-packed-weight-selection/Program.cs.txt').read_text()
    assert source.count(before)==1 and (TOOLS/'Program.cs.txt').read_text()==source.replace(before,after)
    for name in ['vm.py','review.py']:assert (TOOLS/name).read_bytes()==(TOOLS.parent/'owned-packed-weight-selection'/name).read_bytes()
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
