"""Freeze both deployment phases only after real smoke, refusal tests and source commit."""
from pathlib import Path
import argparse,shutil,subprocess,tarfile
from audit import pin,read,write,worker
from protocol import CORE,NATIVE,PROTOCOL,LIMITS,CRITERIA,schedule

ROOT=Path(__file__).resolve().parents[3]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);args=p.parse_args();base=args.artifact.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit frozen tools first'
    smoke=read(base/'smoke-audit.json');state=read(base/'smoke-process/identity.json');assert smoke['passed'] is True and state['complete'] is True and state['code']==0
    assert 'OK' in (base/'analysis-tests-final.log').read_text() and '0 Warning(s)' in (base/'build.log').read_text() and '0 Error(s)' in (base/'build.log').read_text()
    for name,want in smoke['binaries'].items():assert pin(base/'bin'/name)==want,name
    for name in ['Program.cs','Probe.csproj']:assert pin(Path(__file__).with_name(name))==smoke['source'][name],name
    for j in smoke['jobs']:worker(base/'smoke-process'/j['name']/'output',base/'inputs',smoke['model'],smoke['binaries']['FingerprintDeployment.dll'],smoke['binaries']['Microsoft.ML.OnnxRuntime.dll'],smoke['native'],j,j['phase'],True)
    assert smoke['binaries']['Lokad.Onnx.dll']['sha256']==CORE
    qualified=ROOT/'artifacts/e5-fingerprint-product-v2-20260920/closed.json';common=ROOT/'artifacts/e5-fingerprint-balanced-20260920/closed.json'
    assert pin(qualified)['sha256']=='b2631477f8f4b227ececf50ea48fed12f68581d706d56d8dc3c4f268819718a7'
    assert pin(common)['sha256']=='fd23f728c125d07269b0c0403a564a82cdb6ea5a5f96a86343b467e2fcbd908c'
    native=ROOT/'artifacts/e5-public-ort-20260919/bin/libonnxruntime.so';assert pin(native)['sha256']==NATIVE
    model=ROOT/'models/multilingual-e5-small/model.onnx';assert pin(model)==smoke['model']
    payload=base/'payload';payload.mkdir();shutil.copytree(base/'bin',payload/'bin');shutil.copytree(base/'inputs',payload/'inputs')
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ['.cs','.csproj','.py','.md']:shutil.copyfile(path,payload/path.name)
    shutil.copyfile(ROOT/'eng/campaign_processes.py',payload/'campaign_processes.py');shutil.copyfile(ROOT/'global.json',payload/'global.json')
    shutil.copyfile(ROOT/'.agent/m2-fingerprint-deployment-20260920.md',payload/'prospective-plan.md')
    meta=dict(schema=1,protocol=PROTOCOL,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              product_source='faf284489f113504c5631aa4de3515827b6d2e16',qualified_product_receipt=pin(qualified),common_state_receipt=pin(common),
              model=dict(path='/home/vermorel/Onnx/models/multilingual-e5-small/model.onnx',**pin(model)),
              native=dict(path='/home/vermorel/Onnx/artifacts/e5-public-ort-20260919/worker/libonnxruntime.so',**pin(native)),
              limits=LIMITS,criteria=CRITERIA,schedule=schedule(),phases=['aa','compare'],smoke_audit=pin(base/'smoke-audit.json'),analysis_tests=pin(base/'analysis-tests-final.log'),
              files={p.relative_to(payload).as_posix():pin(p) for p in sorted(payload.rglob('*')) if p.is_file()})
    write(payload/'frozen.json',meta)
    with tarfile.open(base/'payload.tar.gz','x:gz') as tar:
        for name in list(meta['files'])+['frozen.json']:tar.add(payload/name,arcname=name,recursive=False)
    write(base/'preparation.json',dict(archive=pin(base/'payload.tar.gz'),frozen=pin(payload/'frozen.json')))
    print('Frozen',len(meta['files']),'files;',pin(base/'payload.tar.gz'))

if __name__=='__main__':main()
