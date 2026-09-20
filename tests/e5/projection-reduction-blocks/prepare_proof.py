"""Freeze the already built prototype; no repeated local execution."""
import argparse, shutil, subprocess, tarfile
from proof_common import *


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args()
    base=args.artifact.resolve();base.mkdir(parents=True,exist_ok=False);lane=Path(__file__).resolve().parent
    local=read(BUILD/'local-closed.json');assert pin(BUILD/'local-closed.json')['sha256']==LOCAL_RECEIPT and local['local_passed']
    for name,wanted in local['files'].items(): assert pin(ROOT/name)==wanted,name
    assert pin(BUILD/'bin/Probe.dll')['sha256']==PROBE and pin(BUILD/'bin/Lokad.Onnx.dll')['sha256']==CORE
    source=read(BUILD/'source/source.json')
    for name,wanted in source['sources'].items():assert pin(ROOT/name)==wanted,name
    for name,wanted in source['files'].items():assert pin(BUILD/'source'/name)==wanted,name
    for name in ['Program.cs','Probe.csproj']:assert pin(lane/name)==source['files'][name]
    subprocess.run(['git','diff','--exit-code','HEAD','--',str(lane)],cwd=ROOT,check=True)
    assert not subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',str(lane)],cwd=ROOT).strip()
    files={}
    for path in (BUILD/'source').iterdir():
        if path.is_file():files['source/'+path.name]=path
    for path in (BUILD/'bin').iterdir():
        if path.suffix in ['.dll','.json']:files['bin/'+path.name]=path
    for path in lane.glob('*.py'):files[path.name]=path
    files['process_support.py']=ROOT/'tests/parakeet/recording-amd/remote.py'
    files['reference/local-closed.json']=BUILD/'local-closed.json'
    files['reference/qualified-product-closed.json']=ROOT/'artifacts/e5-layernorm-product-20260920/closed.json'
    files['plan.md']=ROOT/'.agent/m2-reduction-blocks-20260920.md'
    for name in ['sgemm.cpp','mlasi.h']:files['reference/'+name]=ROOT/'external/onnxruntime/onnxruntime/core/mlas/lib'/name
    payload=base/'payload';payload.mkdir()
    for name,path in files.items():
        target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    bundle=dict(source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),limits=LIMITS,
                cases=cases(),core=CORE,probe=PROBE,files={name:pin(path) for name,path in files.items()},
                origins={name:path.relative_to(ROOT).as_posix() for name,path in files.items()})
    write(payload/'bundle.json',bundle)
    with tarfile.open(base/'payload.tar.gz','x:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file():tar.add(path,arcname=path.relative_to(payload).as_posix(),recursive=False)
    result=dict(archive=pin(base/'payload.tar.gz'),bundle=pin(payload/'bundle.json'),files=len(files))
    write(base/'prepared.json',result);print(json.dumps(result))


if __name__=='__main__':main()
