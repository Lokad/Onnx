"""Freeze smoke-checked probe, qualified product, inputs, scripts and both prospective phases."""
from pathlib import Path
import argparse,shutil,subprocess,tarfile
from audit import CASES,CORE,pin,read,write,worker

ROOT=Path(__file__).resolve().parents[3]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit frozen tools first'
    smoke=read(base/'smoke-audit.json');state=read(base/'smoke-process/identity.json')
    assert smoke['passed'] is True and state['complete'] is True and state['code']==0
    for name,wanted in smoke['binaries'].items():assert pin(base/'bin'/name)==wanted,name
    for name in ['Program.cs','Probe.csproj']:assert pin(Path(__file__).with_name(name))==smoke['source'][name],name
    worker(base/'smoke-process/output',base/'inputs',smoke['model'],smoke['binaries']['FingerprintBalanced.dll'],'smoke',0,0)
    assert smoke['binaries']['Lokad.Onnx.dll']['sha256']==CORE
    assert 'OK' in (base/'analysis-tests.log').read_text()
    assert 'OK' in (base/'closure-tests.log').read_text()
    payload=base/'payload';assert not payload.exists();payload.mkdir()
    shutil.copytree(base/'bin',payload/'bin');shutil.copytree(base/'inputs',payload/'inputs')
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ['.cs','.csproj','.py','.md']:shutil.copyfile(path,payload/path.name)
    shutil.copyfile(ROOT/'eng/campaign_processes.py',payload/'campaign_processes.py')
    shutil.copyfile(ROOT/'global.json',payload/'global.json')
    shutil.copyfile(ROOT/'.agent/m2-fingerprint-balanced-20260920.md',payload/'prospective-plan.md')
    model=ROOT/'models/multilingual-e5-small/model.onnx';assert pin(model)==smoke['model']
    qualified=ROOT/'artifacts/e5-fingerprint-product-v2-20260920/closed.json'
    assert pin(qualified)['sha256']=='b2631477f8f4b227ececf50ea48fed12f68581d706d56d8dc3c4f268819718a7'
    schedule=[dict(name=f'v{visit}-{CASES[index]}',case=CASES[index],case_index=index,visit=visit)
              for visit in range(4) for index in (range(5) if visit%2==0 else reversed(range(5)))]
    frozen=dict(schema=1,protocol='single-graph-fingerprint-local6-v2',source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        qualified_product_receipt=pin(qualified),product_source='faf284489f113504c5631aa4de3515827b6d2e16',model=dict(path='/home/vermorel/Onnx/models/multilingual-e5-small/model.onnx',**pin(model)),
        limits=dict(seconds=300,rss=6*1024**3,available=1024**3),schedule=schedule,phases=['aa','compare'],smoke_audit=pin(base/'smoke-audit.json'),analysis_tests=pin(base/'analysis-tests.log'),closure_tests=pin(base/'closure-tests.log'),
        files={path.relative_to(payload).as_posix():pin(path) for path in sorted(payload.rglob('*')) if path.is_file()})
    write(payload/'frozen.json',frozen)
    archive=base/'payload.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for name in list(frozen['files'])+['frozen.json']:tar.add(payload/name,arcname=name,recursive=False)
    write(base/'preparation.json',dict(archive=pin(archive),frozen=pin(payload/'frozen.json')))
    print('Frozen',len(frozen['files']),'files;',pin(archive))

if __name__=='__main__':main()
