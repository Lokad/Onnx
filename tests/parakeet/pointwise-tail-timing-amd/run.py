"""Freeze one fixed-shape timing comparison of the numerically qualified kernel."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

TOOLS=Path(__file__).resolve().parent;ROOT=TOOLS.parents[2]
sys.path.insert(0,str(TOOLS.parent/'pointwise-tail-contracts-amd'))
loader=importlib.util.spec_from_file_location('original_tail',TOOLS.parent/'pointwise-tail-contracts-amd/run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
pin,read,write=original.pin,original.read,original.write
BASE=ROOT/'artifacts/parakeet-pointwise-tail-timing-amd-20260927'
REMOTE='/dev/shm/lokad-pointwise-tail-timing-20260927'
NUMERICAL=ROOT/'artifacts/parakeet-pointwise-tail-arithmetic-contracts-amd-20260927'
prior=original.prior;transport=original.transport
PRELUDE=original.PRELUDE.replace(original.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def references():
    original.prepared();closed=read(NUMERICAL/'closed.json');assert closed['completed'] and closed['arithmetic_contract_passed']
    assert pin(NUMERICAL/'closed.json')['sha256']=='558f2a523febd0d794bc7da6cdae3381513b7ff3d75dfd4ca4f133f618821cc8'
    for name,wanted in closed['files'].items():assert pin(NUMERICAL/name)==wanted,name
    codegen=read(NUMERICAL/'codegen-review.json');assert codegen['passed'] and codegen['numerical_closure']==pin(NUMERICAL/'closed.json')
    return read(NUMERICAL/'bundle/spec.json')


def prepare():
    assert not BASE.exists();old=references();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    put('contract-source/Program.cs',(TOOLS/'Timing.cs.txt').read_bytes())
    for name in ['TailContracts.csproj','global.json']:put('contract-source/'+name,(NUMERICAL/'bundle/contract-source'/name).read_bytes())
    for name,path in [('remote.py',TOOLS/'vm.py'),('base_build.py',TOOLS.parent/'pointwise-tail-arithmetic-contracts-amd/vm.py'),
        ('parent-audit.py',TOOLS.parent/'pointwise-tail-arithmetic-contracts-amd/audit.py'),('common.py',TOOLS.parent/'managed-phase-amd/remote.py'),
        ('campaign_processes.py',ROOT/'artifacts/parakeet-decoder-lstm-layout-app-amd-20260927/collected/runtime/campaign_processes.py'),
        ('protocol.md',TOOLS/'README.md'),('prospective-plan.md',ROOT/'PLAN.md')]:put(name,path.read_bytes())
    shapes=[dict(m=r['m'],n=r['n'],k=r['k'],control=False) for r in old['raw_cases'] if not r['oracle']]
    assert len(shapes)==38
    shapes += [dict(m=m,n=1024,k=224,control=True) for m in [1024,2048]]
    keys=['boot','prior','feed','external','runtimes','origins','products','compiled_scope','build_limits','capture_limits','minimum_free','output_limit']
    spec={k:old[k] for k in keys}
    jobs=[dict(name=n,role=r) for n,r in [('current-1','baseline'),('candidate-1','candidate'),('candidate-2','candidate'),('current-2','baseline')]]
    spec.update(shapes=shapes,jobs=jobs,warmups=5,measurements=5,repeatability_limit=1.10,regression_limit=1.05,foreign_cpu_limit=.01,
        numerical_closure=pin(NUMERICAL/'closed.json'),codegen_review=pin(NUMERICAL/'codegen-review.json'),
        timing_boundary='Raw packed kernel only; deterministic finite buffers at observed shapes. Packing, destination reset and verification outside the timer.',
        release_admitted=False,no_product_build=True,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),shapes=len(shapes))))


def prepared():
    references();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name
    assert spec['files']['parent-audit.py']==pin(TOOLS.parent/'pointwise-tail-arithmetic-contracts-amd/audit.py')


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    else:
        prepared()
        if action=='stage':transport.stage()
        else:
            kind=sys.argv[2];assert kind in ['build','capture']
            if action=='launch' and kind=='capture':assert read(BASE/'build-review.json')['passed']
            {'launch':transport.launch,'observe':prior.observe,'collect':prior.collect}[action](kind)
