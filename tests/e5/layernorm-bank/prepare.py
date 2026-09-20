"""Stage a small payload bound to existing closed AMD captures, then freeze."""
from pathlib import Path
import argparse,shutil,subprocess,tarfile
from common import CORE,ORIGIN,BANKS,LIMITS,pin,read,write,verify
ROOT=Path(__file__).resolve().parents[3]
OLD=ROOT/'artifacts/e5-layernorm-amd-proof-20260920'

def stage(base):
    assert pin(OLD/'closed.json')['sha256']==ORIGIN and read(OLD/'closed.json')['passed'] is True
    closed=read(OLD/'closed.json');origin=OLD/'collected';dependencies={}
    names=[p.relative_to(origin).as_posix() for p in (origin/'capture').rglob('*') if p.is_file()]
    assert len(names)==511
    names+=['source/Kernels.cs','source/TensorOps.Norm.cs','product/Lokad.Onnx.dll','product/Google.Protobuf.dll']
    for name in names:
        want=closed['files']['collected/'+name];assert pin(origin/name)==want;dependencies[name]=want
    payload=base/'payload';(payload/'source').mkdir(parents=True)
    shutil.copyfile(origin/'source/Kernels.cs',payload/'source/Kernels.cs')
    shutil.copyfile(OLD/'closed.json',payload/'origin-closed.json')
    shutil.copyfile(OLD/'final-verification.json',payload/'origin-terminal.json')
    write(payload/'banks.json',BANKS);write(base/'origin-files.json',dependencies)
    write(base/'build-inputs.json',{name:pin(Path(__file__).parent/name) for name in ['Program.cs','Probe.csproj']})
    print('Staged',len(dependencies),'existing dependencies without copying captures.')

def freeze(base):
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit tools first'
    payload=base/'payload';assert not (payload/'bundle.json').exists()
    for name,want in read(base/'build-inputs.json').items():assert pin(Path(__file__).parent/name)==want
    assert pin(payload/'bin/Lokad.Onnx.dll')['sha256']==CORE
    local=read(base/'local-check.json');assert local['passed'] and local['probe']==pin(payload/'bin/LayerNormBank.dll')
    assert 'OK' in (base/'unit-tests.log').read_text() and '0 Warning(s)' in (base/'build.log').read_text() and '0 Error(s)' in (base/'build.log').read_text()
    (payload/'tools').mkdir()
    for source in Path(__file__).parent.iterdir():
        if source.is_file() and source.suffix in ['.py','.cs','.csproj','.md']:shutil.copyfile(source,payload/'tools'/source.name)
    shutil.copyfile(ROOT/'eng/campaign_processes.py',payload/'tools/campaign_processes.py')
    shutil.copyfile(ROOT/'.agent/m2-layernorm-bank-20260920.md',payload/'prospective-plan.md')
    meta=dict(schema=1,protocol='layernorm-complete-bank-v1',source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        core=CORE,origin_sha256=ORIGIN,limits=LIMITS,banks=BANKS,origin_files=read(base/'origin-files.json'),
        files={p.relative_to(payload).as_posix():pin(p) for p in sorted(payload.rglob('*')) if p.is_file()})
    write(payload/'bundle.json',meta);verify(payload,OLD/'collected')
    with tarfile.open(base/'payload.tar.gz','x:gz') as tar:
        for name in list(meta['files'])+['bundle.json']:tar.add(payload/name,arcname=name,recursive=False)
    write(base/'frozen.json',dict(bundle=pin(payload/'bundle.json'),archive=pin(base/'payload.tar.gz'),files=len(meta['files'])))
    print('Frozen',len(meta['files']),'files;',pin(base/'payload.tar.gz'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['stage','freeze']);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args();globals()[args.action](args.artifact.resolve())
