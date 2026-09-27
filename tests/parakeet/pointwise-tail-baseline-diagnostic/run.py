"""Run the existing raw consumer with the exact baseline in both load contexts."""
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
BASE=ROOT/'artifacts/parakeet-pointwise-tail-baseline-diagnostic-20260927'
REMOTE='/dev/shm/lokad-pointwise-tail-baseline-diagnostic-20260927'
DIAG=ROOT/'artifacts/parakeet-pointwise-tail-nan-diagnostic-20260927'
prior=original.prior;transport=original.transport
PRELUDE=original.PRELUDE.replace(original.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def prepare():
    original.prepared();assert not BASE.exists()
    old=read(original.BASE/'bundle/spec.json');built=read(original.BASE/'build-collected/built.json')
    review=read(DIAG/'review.json');assert review['diagnostic_complete'] and not review['case_differences']
    baseline=built['products']['baseline'];assert baseline['Lokad.Onnx.dll']['sha256']=='47984318b082710c3a4f57a85b1500d49d7c1236c04b1234477d19e48d11207c'
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
    for name,wanted in old['runtime'].items():
        path=original.BASE/'build-collected/runtime/baseline'/name;assert pin(path)==wanted;put('runtime/'+name,path.read_bytes())
    for suffix in ['dll','deps.json','runtimeconfig.json']:
        name='TailContracts.'+suffix;path=DIAG/'capture-collected/runtime'/name
        assert pin(path)==read(DIAG/'capture-collected/built.json')['runtime'][name]
        put('runtime/'+name,path.read_bytes())
    for name,path in [('remote.py',TOOLS/'vm.py'),('common.py',TOOLS.parent/'managed-phase-amd/remote.py'),('protocol.md',TOOLS/'README.md')]:put(name,path.read_bytes())
    write(bundle/'built.json',dict(products=dict(baseline=baseline,candidate=baseline),diagnostic_only=True,no_product_build=True))
    keys=['boot','prior','external','raw_cases','fallback_cases','disasm','capture_limits','minimum_free','output_limit']
    spec={k:old[k] for k in keys}
    spec.update(modes=['normal','avx512-disabled'],baseline=baseline,consumer=pin(bundle/'runtime/TailContracts.dll'),
        original_consumer=pin(DIAG/'capture-collected/runtime/TailContracts.dll'),diagnostic_review=pin(DIAG/'review.json'),
        identical_products=True,no_product_build=True,no_consumer_build=True,no_performance_measurement=True,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'))))


def prepared():
    original.prepared();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    assert spec['diagnostic_review']==pin(DIAG/'review.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


def review():
    prepared();folder=BASE/'capture-collected';receipt=read(folder/'capture-collection.json')
    assert receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    assert read(BASE/'capture-transfer.json')['collection']==pin(folder/'capture-collection.json')
    assert pin(folder/'spec.json')==pin(BASE/'bundle/spec.json')
    spec=read(folder/'spec.json');state=read(folder/'capture-state.json');built=read(folder/'built.json')
    assert built['products']['baseline']==built['products']['candidate']==spec['baseline']
    assert state['supervisor']==read(BASE/'capture-deployment.json') and state['complete'] and state['code']==0
    assert [r['name'] for r in state['runs']]==spec['modes']
    failures={};resources=[]
    for job in state['runs']:
        mode=job['name'];result=read(folder/'probe'/mode/'result.json')
        assert job['complete'] and job['code']==0 and result['completed']
        assert result['pid']==job['owner']['pid'] and result['runtime']=='.NET 10.0.8'
        assert result['core_sha256']==spec['baseline']['Lokad.Onnx.dll']['sha256']
        assert result['consumer_sha256']==spec['consumer']['sha256']
        flags=dict(DOTNET_JitDisasm=spec['disasm'])
        if mode=='avx512-disabled':flags['DOTNET_EnableAVX512']='0'
        assert result['flags']==flags
        failures[mode]=[]
        for wanted,row in zip(spec['raw_cases'],result['results'],strict=True):
            assert {k:row[k] for k in wanted}==wanted
            if row['bit_exact']:assert row['inputs_immutable'] and row['guards_intact'] and row['allocated_bytes']==0
            else:failures[mode].append(row)
        assert len(result['results'])==2598 and result['failed']==len(failures[mode]) and result['passed']==(not failures[mode])
        samples=[json.loads(line) for line in (folder/'logs'/(mode+'.resources.jsonl')).read_text().splitlines()]
        assert len(samples)==job['samples']>0
        for s in samples:
            assert s['rss']<spec['capture_limits']['rss'] and s['seconds']<spec['capture_limits']['seconds']
            assert min(s['available'],s['tmpfs'])>=spec['minimum_free'] and s['output']<spec['output_limit']
            assert all(m['affinity']==[2] and all(t==[2] for t in m['threads']) for m in s['members'])
        resources.append(dict(mode=mode,seconds=job['seconds'],samples=len(samples),peak_rss=max(s['rss'] for s in samples)))
    value=dict(diagnostic_complete=True,identical_products=True,products=built['products'],failures=failures,resources=resources,
        no_performance_measurement=True,release_admitted=False,collection=pin(folder/'capture-collection.json'),reviewer=pin(Path(__file__)))
    write(BASE/'review.json',value)
    print(json.dumps(dict(review=pin(BASE/'review.json'),failures={k:len(v) for k,v in failures.items()},resources=resources)))


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='review':review()
    else:
        prepared()
        if action=='stage':transport.stage()
        else:
            assert sys.argv[2]=='capture'
            {'launch':transport.launch,'observe':prior.observe,'collect':prior.collect}[action]('capture')
