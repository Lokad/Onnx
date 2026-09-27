"""Qualify arithmetic with explicit NaN telemetry after the baseline A/A diagnosis."""
import ast
import difflib
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
BASE=ROOT/'artifacts/parakeet-pointwise-tail-arithmetic-contracts-amd-20260927'
REMOTE='/dev/shm/lokad-pointwise-tail-arithmetic-contracts-20260927'
AA=ROOT/'artifacts/parakeet-pointwise-tail-baseline-diagnostic-20260927'
DIAG=ROOT/'artifacts/parakeet-pointwise-tail-nan-diagnostic-20260927'
prior=original.prior;transport=original.transport
PRELUDE=original.PRELUDE.replace(original.REMOTE,REMOTE)
prior.BASE,prior.REMOTE,prior.PRELUDE=BASE,REMOTE,PRELUDE
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def prepare():
    original.prepared();assert not BASE.exists()
    aa=read(AA/'review.json');assert aa['diagnostic_complete'] and aa['identical_products']
    assert {k:len(v) for k,v in aa['failures'].items()}==dict(normal=214,**{'avx512-disabled':218})
    old=read(original.BASE/'bundle/spec.json');built=read(original.BASE/'build-collected/built.json')
    before=(DIAG/'bundle/contract-source/Program.cs').read_text(encoding='utf8');after=before
    start=after.index('    static void Same(');end=after.index('    [MethodImpl(',start)
    after=after[:start]+'    static int Same(ReadOnlySpan<float> a, ReadOnlySpan<float> b, string reason) => ArithmeticComparison.Same(a, b, reason);\n'+after[end:]
    def change(a,b,count=1):
        nonlocal after
        assert after.count(a)==count,a;after=after.replace(a,b)
    change('long allocated;','long allocated; int nanPayloadDifferences = 0;')
    change('Same(expected,actual,','nanPayloadDifferences += Same(expected,actual,',2)
    change('return new { m,n,k,exceptional,oracle,bit_exact=true,inputs_immutable=true,guards_intact=true,allocated_bytes=allocated,output_sha256=Hash(actual) };',
        'return new { m,n,k,exceptional,oracle,arithmetic_contract_passed=true,non_nan_bits_exact=true,nan_classification_exact=true,nan_payload_differences=nanPayloadDifferences,bit_exact=nanPayloadDifferences==0,inputs_immutable=true,guards_intact=true,allocated_bytes=allocated,output_sha256=Hash(actual),baseline_output_sha256=Hash(expected) };')
    change('Require(args.Length==4,"spec built output mode");','Require(args.Length==4,"spec built output mode");\n        ArithmeticComparison.Verify();')
    change('new {m,n,k,exceptional,oracle,bit_exact=false,error=error.ToString()}',
        'new {m,n,k,exceptional,oracle,arithmetic_contract_passed=false,bit_exact=false,error=error.ToString()}')
    change('new {completed=true,passed=failed==0,failed,mode,pid=Environment.ProcessId,',
        'new {completed=true,passed=failed==0,failed,mode,comparison_selftests=ArithmeticComparison.SelfTestCount,pid=Environment.ProcessId,')
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        p=bundle/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    put('contract-source/Program.cs',after.encode())
    put('contract-source/ArithmeticComparison.cs',(TOOLS/'ArithmeticComparison.cs.txt').read_bytes())
    for name in ['TailContracts.csproj','global.json']:put('contract-source/'+name,(DIAG/'bundle/contract-source'/name).read_bytes())
    put('consumer.patch',''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='original-consumer',tofile='arithmetic-consumer')).encode())
    for name,path in [('remote.py',TOOLS/'vm.py'),('common.py',TOOLS.parent/'managed-phase-amd/remote.py'),('protocol.md',TOOLS/'README.md'),('prospective-plan.md',ROOT/'PLAN.md')]:put(name,path.read_bytes())
    runtimes={role:{n.removeprefix(role+'/'):v for n,v in built['runtime'].items() if n.startswith(role+'/') and not n.startswith(role+'/TailContracts.')} for role in ['baseline','candidate']}
    assert runtimes['candidate']['Lokad.Onnx.dll']['sha256']=='7cac67880fa9a4d519ac18e5887f47f48f0f14903bdf74cc6561b45c851e4f27'
    origins={role:original.REMOTE+'/runtime/'+role for role in runtimes}
    external={origins[role]+'/'+n:v for role,files in runtimes.items() for n,v in files.items()}
    external.update(old['external'])
    keys=['boot','prior','feed','raw_cases','fallback_cases','disasm','build_limits','capture_limits','minimum_free','output_limit']
    spec={k:old[k] for k in keys}
    jobs=[dict(name=role+'-'+mode,role=role,mode=mode) for role in ['baseline','candidate'] for mode in ['normal','avx512-disabled']]
    jobs += [dict(name='scalar-'+role,role=role,mode='scalar-'+role) for role in ['baseline','candidate']]
    spec.update(external=external,runtimes=runtimes,origins=origins,products=built['products'],jobs=jobs,
        baseline_diagnosis=pin(AA/'review.json'),original_failed_closure=pin(original.BASE/'closed.json'),compiled_scope=pin(original.BASE/'build-review.json'),
        consumer_before=pin(DIAG/'bundle/contract-source/Program.cs'),source=old['source'],no_product_build=True,no_performance_measurement=True,release_admitted=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    for p in TOOLS.glob('*.py'):ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'))))


def prepared():
    original.prepared();value=read(BASE/'prepared.json');spec=read(BASE/'bundle/spec.json')
    assert value['archive']==pin(BASE/'payload.tar.gz') and value['spec']==pin(BASE/'bundle/spec.json')
    assert spec['baseline_diagnosis']==pin(AA/'review.json') and spec['compiled_scope']==pin(original.BASE/'build-review.json')
    for name,wanted in value['tools'].items():assert pin(TOOLS/name)==wanted,name
    for name,wanted in spec['files'].items():assert pin(BASE/'bundle'/name)==wanted,name


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
