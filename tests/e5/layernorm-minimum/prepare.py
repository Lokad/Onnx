"""Stage and freeze minimum-work conditioning from the closed v2 producer."""
from pathlib import Path
import argparse,difflib,hashlib,json,shutil,subprocess,sys,tarfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
PRIOR=ROOT/'artifacts/e5-layernorm-conditioned-20260920'
OLD=ROOT/'artifacts/e5-layernorm-amd-proof-20260920'
PRIOR_SHA='500fe4075d7bdbf99362990e493978e4c88f75d046e6aa1efcd329f440e70b3b'
ORIGIN='96ec37f6a9355a86d429822605dcc568b65bfaef870d4f61384dcacbe188a630'
PROTOCOL='layernorm-minimum-bank-v3'

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
    if name in ['Program.cs','common.py','timing.py','test_audit.py']:text=replace(text,'layernorm-conditioned-bank-v2',PROTOCOL)
    if name=='Program.cs':
        text=replace(text,'conditioningTicks < 3 * Stopwatch.Frequency','conditioningTicks < 3 * Stopwatch.Frequency || warmCycle < 128')
        text=replace(text,'conditioning_seconds = 3','conditioning_seconds = 3, conditioning_min_cycles = 128')
    elif name=='common.py':
        text=replace(text,"REMOTE='/home/vermorel/Onnx/artifacts/e5-layernorm-conditioned-20260920'","REMOTE='/home/vermorel/Onnx/artifacts/e5-layernorm-minimum-20260920'")
        text=replace(text,"    old=read(base/'origin-closed.json')",f"    assert pin(base/'conditioned-closed.json')['sha256']=='{PRIOR_SHA}'\n    old=read(base/'origin-closed.json')")
    elif name=='timing.py':
        text=replace(text,"assert len(warm)>0 and len(warm)%4==0","assert len(warm)>=128*4 and len(warm)%4==0")
        text=replace(text,"assert case['conditioning_seconds']==3","assert case['conditioning_seconds']==3 and case['conditioning_min_cycles']==128")
        text=replace(text,"assert total-sum(s['ticks'] for s in warm[-4:])<3*frequency,","assert len(warm)//4==128 or total-sum(s['ticks'] for s in warm[-4:])<3*frequency,")
    elif name=='test_audit.py':
        old="(3_000_000_000+3_950_000*d['repeats']-1)//(3_950_000*d['repeats'])"
        text=replace(text,old,'max(128,'+old+')')
        text=replace(text,"case['conditioning_seconds']=3","case['conditioning_seconds']=3;case['conditioning_min_cycles']=128")
        text=section(text,'    def test_conditioning_boundary_and_no_extra_cycles(self):','    def test_resource_guards_and_births(self):',(HERE/'test_conditioning.py.txt').read_text()+'\n')
    elif name=='vm.py':
        marker="    script=prefix()+'births=%r\\ndependencies=%r\\n'%(births,bundle['origin_files'])"
        extra=f"""    conditioned=ROOT/'artifacts/e5-layernorm-conditioned-20260920';conditioned_closed=read(conditioned/'closed.json')
    assert pin(conditioned/'closed.json')['sha256']=='{PRIOR_SHA}' and conditioned_closed['execution_passed'] and not conditioned_closed['performance_passed']
    for name,want in conditioned_closed['files'].items():assert pin(conditioned/name)==want,name
    for name,want in conditioned_closed['reports'].items():assert pin(Path(name))==want,name
    conditioned_final=read(conditioned/'final-verification.json');assert conditioned_final['passed'] and conditioned_final['closed']==pin(conditioned/'closed.json')
    births+=conditioned_final['terminal']['births'];eligibility['conditioned_closed']=pin(conditioned/'closed.json')
"""
        text=replace(text,marker,extra+marker)
    elif name=='close.py':
        text=replace(text,"assert len(warm)>0 and len(warm)%4==0 and case['conditioning_seconds']==3","assert len(warm)>=128*4 and len(warm)%4==0 and case['conditioning_seconds']==3 and case['conditioning_min_cycles']==128")
        text=replace(text,"and total-sum(r['ticks'] for r in warm[-4:])<threshold","and (len(warm)//4==128 or total-sum(r['ticks'] for r in warm[-4:])<threshold)")
    elif name=='report.py':
        text=replace(text,"directory=root/'tests/e5/layernorm-conditioned'","directory=root/'tests/e5/layernorm-minimum'")
        text=replace(text,'# Conditioned complete LayerNorm banks','# Complete LayerNorm banks after minimum-work conditioning')
        text=replace(text,'This distinct protocol follows the [inconclusive fixed-warmup result](../layernorm-bank/results-20260920.md). It preserves all nine banks, four fresh sequential workers and every original control/gain/regression threshold. Only prospective conditioning, batch size and measured cycle count change; no old result is relabeled.',
            'This distinct protocol follows the [rejected time-only conditioning result](../layernorm-conditioned/results-20260920.md). It preserves all nine banks, four fresh workers, kernel sources, measured schedule and every original threshold. Only conditioning gains a fixed minimum of 128 complete four-role cycles. The earlier rejection remains unchanged.')
        text=replace(text,'Conditioning stops after the first complete four-role cycle whose summed kernel time reaches three seconds.','Conditioning stops after the first complete four-role cycle where both the summed kernel time reaches three seconds and at least 128 cycles have completed.')
        text=replace(text,'This fixed work-time budget is not convergence selection or proof of a particular JIT tier.','These fixed time and minimum-work budgets are not convergence selection or proof of a particular JIT tier.')
        text=replace(text,'artifacts/e5-layernorm-conditioned-20260920','artifacts/e5-layernorm-minimum-20260920')
        text=replace(text,'The preceding failed controls remain unchanged.','The preceding rejection and earlier failed controls remain unchanged.')
    return text

def stage(base):
    assert not base.exists(),'Single-use stage'
    assert pin(PRIOR/'closed.json')['sha256']==PRIOR_SHA and pin(OLD/'closed.json')['sha256']==ORIGIN
    prior=read(PRIOR/'closed.json');assert prior['execution_passed'] and not prior['performance_passed']
    origin=OLD/'collected';closed=read(OLD/'closed.json');old_bundle=read(PRIOR/'collected/bundle.json')
    assert pin(PRIOR/'collected/bundle.json')==prior['files']['collected/bundle.json']
    dependencies=old_bundle['origin_files'];assert len(dependencies)==515
    for name,want in dependencies.items():assert pin(origin/name)==want==closed['files']['collected/'+name],name
    payload=base/'payload';(payload/'source').mkdir(parents=True);(base/'tools').mkdir();(base/'host').mkdir()
    shutil.copyfile(origin/'source/Kernels.cs',payload/'source/Kernels.cs')
    for name in ['origin-closed.json','origin-terminal.json','prior-closed.json','prior-terminal.json']:
        source=PRIOR/'collected'/name;assert pin(source)==prior['files']['collected/'+name];shutil.copyfile(source,payload/name)
    shutil.copyfile(PRIOR/'closed.json',payload/'conditioned-closed.json');shutil.copyfile(PRIOR/'final-verification.json',payload/'conditioned-terminal.json')
    names=['Program.cs','Probe.csproj','common.py','data.py','local_check.py','run.py','vm.py','audit.py','timing.py','report.py','test_audit.py','close.py','campaign_processes.py']
    sources={};generated={};diff=[]
    for name in names:
        host=name.endswith(('.cs','.csproj'));key='collected/'+('source/' if host else 'tools/')+name;source=PRIOR/key
        sources[key]=pin(source);assert sources[key]==prior['files'][key]
        before=source.read_text();after=transform(name,before);target=base/('host' if host else 'tools')/name
        if before==after:shutil.copyfile(source,target)
        else:
            with target.open('x',encoding='utf-8',newline='\n') as stream:stream.write(after)
        generated[target.relative_to(base).as_posix()]=pin(target)
        diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='conditioned/'+name,tofile='minimum/'+name))
    with (base/'generated.diff').open('x',encoding='utf-8') as stream:stream.write(''.join(diff))
    write(base/'generation.json',dict(prior=pin(PRIOR/'closed.json'),sources=sources,generated=generated,diff=pin(base/'generated.diff'),generator={p.name:pin(p) for p in HERE.iterdir() if p.is_file()}))
    sys.path.insert(0,str(base/'tools'));from common import BANKS
    write(payload/'banks.json',BANKS);write(base/'origin-files.json',dependencies)
    write(base/'build-inputs.json',{name:pin(base/'host'/name) for name in ['Program.cs','Probe.csproj']})
    print('Generated',len(generated),'pinned files; unchanged',len(dependencies),'origin dependencies.')

def freeze(base):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit tools first'
    payload=base/'payload';assert not (payload/'bundle.json').exists();generation=read(base/'generation.json')
    for name,want in generation['generator'].items():assert pin(HERE/name)==want,name
    for name,want in generation['generated'].items():assert pin(base/name)==want,name
    for name,want in read(base/'build-inputs.json').items():assert pin(base/'host'/name)==want,name
    assert pin(base/'generated.diff')==generation['diff']
    sys.path.insert(0,str(base/'tools'));from common import CORE,BANKS,LIMITS,verify
    assert pin(payload/'bin/Lokad.Onnx.dll')['sha256']==CORE
    local=read(base/'local-check.json');assert local['passed'] and local['probe']==pin(payload/'bin/LayerNormBank.dll')
    assert 'OK' in (base/'unit-tests.log').read_text() and '0 Warning(s)' in (base/'build.log').read_text() and '0 Error(s)' in (base/'build.log').read_text()
    shutil.copytree(base/'tools',payload/'tools');(payload/'generator').mkdir()
    for name in generation['generator']:shutil.copyfile(HERE/name,payload/'generator'/name)
    for name in ['Program.cs','Probe.csproj']:shutil.copyfile(base/'host'/name,payload/'source'/name)
    for name in ['generation.json','generated.diff','build-inputs.json']:shutil.copyfile(base/name,payload/name)
    shutil.copyfile(ROOT/'.agent/m2-layernorm-minimum-20260920.md',payload/'prospective-plan.md')
    meta=dict(schema=1,protocol=PROTOCOL,source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),core=CORE,origin_sha256=ORIGIN,
        limits=LIMITS,banks=BANKS,origin_files=read(base/'origin-files.json'),files={p.relative_to(payload).as_posix():pin(p) for p in sorted(payload.rglob('*')) if p.is_file()})
    write(payload/'bundle.json',meta);verify(payload,origin=OLD/'collected')
    with tarfile.open(base/'payload.tar.gz','x:gz') as tar:
        for name in list(meta['files'])+['bundle.json']:tar.add(payload/name,arcname=name,recursive=False)
    with tarfile.open(base/'payload.tar.gz') as tar:
        members=tar.getmembers();assert len(members)==len(meta['files'])+1 and {m.name for m in members}==set(meta['files'])|{'bundle.json'}
        for member in members:
            with tar.extractfile(member) as stream:actual=dict(bytes=member.size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
            assert actual==pin(payload/member.name),member.name
    write(base/'frozen.json',dict(bundle=pin(payload/'bundle.json'),archive=pin(base/'payload.tar.gz'),files=len(meta['files']),archive_verified=True))
    print('Frozen and verified',len(meta['files']),'payload files;',pin(base/'payload.tar.gz'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['stage','freeze']);parser.add_argument('--artifact',type=Path,required=True)
    args=parser.parse_args();globals()[args.action](args.artifact.resolve())
