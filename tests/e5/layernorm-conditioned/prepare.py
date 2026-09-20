"""Generate a declared protocol revision from receipt-bound v1 sources."""
from pathlib import Path
import argparse,difflib,hashlib,json,shutil,subprocess,sys,tarfile

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PRIOR=ROOT/'artifacts/e5-layernorm-bank-20260920'
OLD=ROOT/'artifacts/e5-layernorm-amd-proof-20260920'
PRIOR_SHA='40a0cfd7e39e2708462574f160c2df455848a062d2d5597bd03c2d035cfb6d64'
ORIGIN='96ec37f6a9355a86d429822605dcc568b65bfaef870d4f61384dcacbe188a630'
PROTOCOL='layernorm-conditioned-bank-v2'

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))
def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)
def replace(text,old,new):
    assert text.count(old)==1,(old,text.count(old))
    return text.replace(old,new)
def section(text,start,end,new):
    assert text.count(start)==text.count(end)==1
    a=text.index(start);b=text.index(end,a);return text[:a]+new+text[b:]

def transform(name,text):
    if name=='Program.cs':
        text=section(text,'            var warmup = new Sample[64];','            Require(Hash(held)',(HERE/'conditioning.cs.txt').read_text())
        text=replace(text,'first, warmup, measured };','first, warmup, measured, conditioning_ticks = conditioningTicks, conditioning_seconds = 3 };')
        text=replace(text,'layernorm-complete-bank-v1',PROTOCOL)
    elif name=='common.py':
        text=replace(text,"REMOTE='/home/vermorel/Onnx/artifacts/e5-layernorm-bank-20260920'","REMOTE='/home/vermorel/Onnx/artifacts/e5-layernorm-conditioned-20260920'")
        text=replace(text,'[128,32,8,8,2]','[512,128,32,32,8]')
        text=replace(text,'diagnostic=True,repeats=32','diagnostic=True,repeats=128')
        text=replace(text,'layernorm-complete-bank-v1',PROTOCOL)
        text=replace(text,"    old=read(base/'origin-closed.json')",f"    assert pin(base/'prior-closed.json')['sha256']=='{PRIOR_SHA}'\n    old=read(base/'origin-closed.json')")
    elif name=='audit.py':
        text=section(text,'def inspect_timing(', 'def summarize(', 'from timing import inspect_timing\n\n\n')
        text=replace(text,'rows=[];telemetry=[];','warmup_count=0;rows=[];telemetry=[];')
        marker="        current=[c['output_sha256'] for c in value['results']]"
        text=replace(text,marker,"        warmup_count+=sum(len(c['warmup']) for c in value['results'])\n"+marker)
        text=replace(text,'measured_samples=6912,warmup_samples=2304','measured_samples=13824,warmup_samples=warmup_count')
    elif name=='test_audit.py':
        text=replace(text,"[('first',1),('warmup',16),('measured',48)]","[('first',1),('warmup',(3_000_000_000+3_950_000*d['repeats']-1)//(3_950_000*d['repeats'])),('measured',96)]")
        text=replace(text,'            cases.append(case)',"            case['conditioning_ticks']=sum(r['ticks'] for r in case['warmup']);case['conditioning_seconds']=3\n            cases.append(case)")
        text=replace(text,'layernorm-complete-bank-v1',PROTOCOL)
        extra=(HERE/'test_conditioning.py.txt').read_text()
        text=replace(text,'    def test_resource_guards_and_births(self):',extra+'\n    def test_resource_guards_and_births(self):')
    elif name=='vm.py':
        marker="    script=prefix()+'births=%r\\ndependencies=%r\\n'%(births,bundle['origin_files'])"
        extra=f"""    previous=ROOT/'artifacts/e5-layernorm-bank-20260920';previous_closed=read(previous/'closed.json')
    assert pin(previous/'closed.json')['sha256']=='{PRIOR_SHA}' and previous_closed['execution_passed'] and not previous_closed['performance_passed']
    for name,want in previous_closed['files'].items():assert pin(previous/name)==want,name
    for name,want in previous_closed['reports'].items():assert pin(Path(name))==want,name
    previous_final=read(previous/'final-verification.json');assert previous_final['passed'] and previous_final['closed']==pin(previous/'closed.json')
    births+=previous_final['terminal']['births'];eligibility['previous_closed']=pin(previous/'closed.json')
"""
        text=replace(text,marker,extra+marker)
    elif name=='close.py':
        text=replace(text,"            for label in counts:counts[label]+=len(case[label])","""            warm=case['warmup'];assert len(warm)>0 and len(warm)%4==0 and case['conditioning_seconds']==3
            total=sum(r['ticks'] for r in warm);threshold=3*timing['frequency']
            assert total==case['conditioning_ticks'] and total>=threshold and total-sum(r['ticks'] for r in warm[-4:])<threshold
            for label in counts:counts[label]+=len(case[label])""")
        text=replace(text,'assert len(records)==48','assert len(records)==96')
        text=replace(text,'dict(measured=6912,warmup=2304,first=144)',"dict(measured=13824,warmup=audit['warmup_samples'],first=144)")
        text=section(text,"    report=Path(__file__).with_name('results-20260920.md');",'    files={p.relative_to(base)',"    from report import render\n    reports=render(base,payload,audit)\n")
        text=replace(text,'for p in [report,observations]','for p in reports')
    return text

def stage(base):
    assert not base.exists(),'A stage is single use; preserve failed attempts'
    assert pin(PRIOR/'closed.json')['sha256']==PRIOR_SHA
    assert pin(OLD/'closed.json')['sha256']==ORIGIN and read(OLD/'closed.json')['passed']
    previous=read(PRIOR/'closed.json');closed=read(OLD/'closed.json');origin=OLD/'collected'
    assert previous['execution_passed'] and not previous['performance_passed']
    names=[p.relative_to(origin).as_posix() for p in (origin/'capture').rglob('*') if p.is_file()]
    assert len(names)==511
    names+=['source/Kernels.cs','source/TensorOps.Norm.cs','product/Lokad.Onnx.dll','product/Google.Protobuf.dll']
    dependencies={}
    for name in names:
        want=closed['files']['collected/'+name];assert pin(origin/name)==want;dependencies[name]=want
    payload=base/'payload';(payload/'source').mkdir(parents=True);(base/'tools').mkdir();(base/'host').mkdir()
    shutil.copyfile(origin/'source/Kernels.cs',payload/'source/Kernels.cs')
    for source,name in [(OLD/'closed.json','origin-closed.json'),(OLD/'final-verification.json','origin-terminal.json'),(PRIOR/'closed.json','prior-closed.json'),(PRIOR/'final-verification.json','prior-terminal.json')]:shutil.copyfile(source,payload/name)
    source_files=['Program.cs','Probe.csproj','common.py','data.py','local_check.py','run.py','vm.py','audit.py','test_audit.py','close.py','campaign_processes.py']
    sources={};generated={};diff=[]
    for name in source_files:
        source=PRIOR/'collected/tools'/name;want=previous['files']['collected/tools/'+name];assert pin(source)==want
        sources[name]=want;before=source.read_text();after=transform(name,before)
        target=base/('host' if name.endswith(('.cs','.csproj')) else 'tools')/name
        # Preserve bytes exactly for unmodified dependencies.
        if before==after:shutil.copyfile(source,target)
        else:
            with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(after)
        generated[target.relative_to(base).as_posix()]=pin(target)
        diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='prior/'+name,tofile='conditioned/'+name))
    for name in ['timing.py','report.py']:
        target=base/'tools'/name;shutil.copyfile(HERE/name,target);generated[target.relative_to(base).as_posix()]=pin(target)
    with (base/'generated.diff').open('x',encoding='utf-8') as stream:stream.write(''.join(diff))
    write(base/'generation.json',dict(prior=pin(PRIOR/'closed.json'),sources=sources,generated=generated,diff=pin(base/'generated.diff'),
        generator={p.name:pin(p) for p in HERE.iterdir() if p.is_file()}))
    sys.path.insert(0,str(base/'tools'));from common import BANKS
    write(payload/'banks.json',BANKS);write(base/'origin-files.json',dependencies)
    write(base/'build-inputs.json',{name:pin(base/'host'/name) for name in ['Program.cs','Probe.csproj']})
    print('Generated',len(generated),'files from pinned source; bound',len(dependencies),'existing dependencies.')

def freeze(base):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit tools first'
    payload=base/'payload';assert not (payload/'bundle.json').exists()
    generation=read(base/'generation.json')
    for name,want in generation['generator'].items():assert pin(HERE/name)==want,name
    for name,want in generation['generated'].items():assert pin(base/name)==want,name
    assert pin(base/'generated.diff')==generation['diff']
    for name,want in read(base/'build-inputs.json').items():assert pin(base/'host'/name)==want
    sys.path.insert(0,str(base/'tools'));from common import CORE,BANKS,LIMITS,verify
    assert pin(payload/'bin/Lokad.Onnx.dll')['sha256']==CORE
    local=read(base/'local-check.json');assert local['passed'] and local['probe']==pin(payload/'bin/LayerNormBank.dll')
    assert 'OK' in (base/'unit-tests.log').read_text() and '0 Warning(s)' in (base/'build.log').read_text() and '0 Error(s)' in (base/'build.log').read_text()
    shutil.copytree(base/'tools',payload/'tools');(payload/'generator').mkdir()
    for name in generation['generator']:shutil.copyfile(HERE/name,payload/'generator'/name)
    for name in ['Program.cs','Probe.csproj']:shutil.copyfile(base/'host'/name,payload/'source'/name)
    for name in ['generation.json','generated.diff','build-inputs.json']:shutil.copyfile(base/name,payload/name)
    shutil.copyfile(ROOT/'.agent/m2-layernorm-conditioned-20260920.md',payload/'prospective-plan.md')
    meta=dict(schema=1,protocol=PROTOCOL,source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        core=CORE,origin_sha256=ORIGIN,limits=LIMITS,banks=BANKS,origin_files=read(base/'origin-files.json'),
        files={p.relative_to(payload).as_posix():pin(p) for p in sorted(payload.rglob('*')) if p.is_file()})
    write(payload/'bundle.json',meta);verify(payload,OLD/'collected')
    with tarfile.open(base/'payload.tar.gz','x:gz') as tar:
        for name in list(meta['files'])+['bundle.json']:tar.add(payload/name,arcname=name,recursive=False)
    with tarfile.open(base/'payload.tar.gz') as tar:
        members=tar.getmembers();assert len(members)==len(meta['files'])+1 and {m.name for m in members}==set(meta['files'])|{'bundle.json'}
        for member in members:
            with tar.extractfile(member) as stream:actual=dict(bytes=member.size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
            assert actual==pin(payload/member.name),member.name
    write(base/'frozen.json',dict(bundle=pin(payload/'bundle.json'),archive=pin(base/'payload.tar.gz'),files=len(meta['files']),archive_verified=True))
    print('Frozen and archive-verified',len(meta['files']),'files;',pin(base/'payload.tar.gz'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['stage','freeze']);parser.add_argument('--artifact',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.artifact.resolve())
