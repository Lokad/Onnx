"""Observe the failed candidate's AVX2 code without rebuilding either product."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

TOOLS=Path(__file__).resolve().parent;ROOT=TOOLS.parents[2]
sys.path.insert(0,str(TOOLS.parent/'pointwise-tail-contracts-amd'))
loader=importlib.util.spec_from_file_location('original_tail_contracts',TOOLS.parent/'pointwise-tail-contracts-amd/run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
pin,read,write,ssh=original.pin,original.read,original.write,original.ssh
FAILED=original.BASE
BASE=ROOT/'artifacts/parakeet-pointwise-tail-nan-diagnostic-20260927'
REMOTE='/dev/shm/lokad-pointwise-tail-nan-diagnostic-20260927'
prior=original.prior;transport=original.transport
PRELUDE=original.PRELUDE.replace(original.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def references():
    original.prepared()
    assert pin(FAILED/'closed.json')['sha256']=='58cb40269af7cda8252007bc059f04ccd4e59697d258e3affbbe48329b8208d9'
    closed=read(FAILED/'closed.json');assert closed['completed'] and not closed['passed']
    for name,wanted in closed['files'].items():assert pin(FAILED/name)==wanted,name
    return read(FAILED/'bundle/spec.json'),read(FAILED/'build-collected/built.json')


def prepare():
    assert not BASE.exists();old,built=references()
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    before=(FAILED/'bundle/contract-source/Program.cs').read_bytes()
    old_check=b': flags.Count==1 && flags.GetValueOrDefault("DOTNET_EnableAVX512")=="0","Declared flags");'
    new_check=b': flags.Count==2 && flags.GetValueOrDefault("DOTNET_EnableAVX512")=="0" && flags.GetValueOrDefault("DOTNET_JitDisasm")==spec.GetProperty("disasm").GetString(),"Declared flags");'
    assert before.count(old_check)==1
    after=before.replace(old_check,new_check);assert after.replace(new_check,old_check)==before
    put('contract-source/Program.cs',after)
    for name in ['TailContracts.csproj','global.json']:put('contract-source/'+name,(FAILED/'bundle/contract-source'/name).read_bytes())
    for name,path in [('remote.py',TOOLS/'vm.py'),('common.py',TOOLS.parent/'managed-phase-amd/remote.py'),('protocol.md',TOOLS/'README.md'),('original-built.json',FAILED/'build-collected/built.json')]:put(name,path.read_bytes())
    runtime={n.removeprefix('candidate/'):v for n,v in built['runtime'].items() if n.startswith('candidate/') and not n.startswith('candidate/TailContracts.')}
    candidate=original.REMOTE+'/runtime/candidate'
    external={**old['external'],**{candidate+'/'+n:v for n,v in runtime.items()}}
    keys=['boot','prior','feed','raw_cases','fallback_cases','disasm','build_limits','capture_limits','minimum_free','output_limit']
    spec={k:old[k] for k in keys}
    spec.update(external=external,candidate=candidate,runtime=runtime,failed_closure=pin(FAILED/'closed.json'),
        consumer_before=pin(FAILED/'bundle/contract-source/Program.cs'),consumer_change='Permit only the additional declared JIT disassembly flag in the AVX512-disabled process.',
        diagnostic_only=True,release_admitted=False,no_model_execution=True,no_product_build=True,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'))))


def prepared():
    references();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


def review():
    prepared();folder=BASE/'capture-collected';receipt=read(folder/'capture-collection.json')
    assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert read(BASE/'capture-transfer.json')['collection']==pin(folder/'capture-collection.json')
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    spec=read(folder/'spec.json');built=read(folder/'built.json')
    for name,wanted in built['runtime'].items():assert pin(folder/'runtime'/name)==wanted,name
    for name,wanted in spec['runtime'].items():assert built['runtime'][name]==wanted,name
    state=read(folder/'capture-state.json');assert state['supervisor']==read(BASE/'capture-deployment.json')
    assert state['complete'] and state['code']==0 and len(state['runs'])==1
    row=state['runs'][0];assert row['name']=='avx512-disabled' and row['complete'] and row['code']==0
    result=read(folder/'probe/avx512-disabled/result.json')
    old=read(FAILED/'capture-collected/probe/avx512-disabled/result.json')
    assert result['completed'] and result['core_sha256']==old['core_sha256']
    assert result['consumer_sha256']==built['runtime']['TailContracts.dll']['sha256']
    assert result['pid']==row['owner']['pid'] and result['runtime']==old['runtime']
    assert result['flags']==dict(DOTNET_EnableAVX512='0',DOTNET_JitDisasm=spec['disasm'])
    assert len(result['results'])==len(old['results'])==len(spec['raw_cases'])==2598
    differences=[]
    for wanted,a,b in zip(spec['raw_cases'],old['results'],result['results'],strict=True):
        assert {k:b[k] for k in wanted}==wanted
        # Source line numbers can change in the consumer; retain the actual failure.
        def normalized(r):return {k:(v.splitlines()[0] if k=='error' else v) for k,v in r.items()}
        if normalized(a)!=normalized(b):differences.append(dict(case=wanted,before=a,after=b))
    samples=[json.loads(s) for s in (folder/'logs/avx512-disabled.resources.jsonl').read_text().splitlines()]
    assert len(samples)==row['samples']>0
    for sample in samples:
        assert sample['seconds']<spec['capture_limits']['seconds'] and sample['rss']<spec['capture_limits']['rss']
        assert min(sample['available'],sample['tmpfs'])>=spec['minimum_free'] and sample['output']<spec['output_limit']
        assert all(m['affinity']==[2] and all(t==[2] for t in m['threads']) for m in sample['members'])
    disasm=folder/'logs/avx512-disabled.stdout';text=disasm.read_text()
    assert 'PackedColumnMaskedEightRows' in text and 'mm_unsafe_vectorized_intrinsics_2x4packed_bump' in text
    value=dict(diagnostic_complete=True,failed_before=old['failed'],failed_observed=result['failed'],case_differences=differences,
        original_failure_preserved=spec['failed_closure'],products_unchanged=True,disassembly=pin(disasm),consumer_source=spec['files']['contract-source/Program.cs'],
        resources=dict(samples=len(samples),seconds=row['seconds'],peak_rss=max(s['rss'] for s in samples)),
        no_performance_measurement=True,release_admitted=False,reviewer=pin(Path(__file__)))
    write(BASE/'review.json',value)
    print(json.dumps({k:v for k,v in value.items() if k!='case_differences'}|dict(case_differences=len(differences))))


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='review':review()
    else:
        prepared()
        if action=='stage':transport.stage()
        else:{'launch':transport.launch,'observe':prior.observe,'collect':prior.collect}[action](sys.argv[2])
