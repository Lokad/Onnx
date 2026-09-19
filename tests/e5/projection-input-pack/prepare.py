"""Freeze a built prototype without downloading models or rebuilding product code."""
from pathlib import Path
import argparse,json,hashlib,shutil,subprocess,tarfile

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return dict(bytes=p.stat().st_size,sha256=sha(p))

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True)
    a=p.parse_args();base=a.artifact.resolve();root=Path(__file__).resolve().parents[3];lane=Path(__file__).parent
    payload=base/'payload';assert not payload.exists() and not (base/'payload.tar.gz').exists()
    source=json.loads((base/'source/source.json').read_text())
    for name,value in source['source'].items():assert pin(root/name)==value
    assert pin(lane/'generate.py')==source['generator']
    for name,value in source['files'].items():assert pin(base/'source'/name)==value
    for name in ['Program.cs','Probe.csproj']:assert pin(lane/name)==source['files'][name]
    assert sha(base/'bin/Lokad.Onnx.dll')=='05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2'
    host=json.loads(subprocess.check_output(['dotnet',str(base/'bin/Probe.dll'),'--host'],text=True))
    assert host==dict(Avx512F=False,Fma=True,unsupported_refusal_checked=True,packing_checks=54)
    files={}
    for f in (base/'source').iterdir():
        if f.is_file():files['source/'+f.name]=f
    for f in (base/'bin').iterdir():
        if f.suffix in ('.dll','.json'):files['bin/'+f.name]=f
    for f in lane.glob('*.py'):files[f.name]=f
    files['process_support.py']=root/'tests/parakeet/recording-amd/remote.py'
    files['campaign_processes.py']=root/'eng/campaign_processes.py'
    files['plan.md']=root/'.agent/m2-projection-input-pack-20260919.md'
    payload.mkdir()
    for name,path in files.items():
        target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    bundle=dict(schema=1,source_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),host=host,
        files={name:pin(path) for name,path in files.items()},local_sources={name:path.relative_to(root).as_posix() for name,path in files.items()})
    (payload/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n',encoding='utf-8')
    archive=base/'payload.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file():tar.add(path,arcname=path.relative_to(payload).as_posix(),recursive=False)
    record=dict(**pin(archive),bundle_sha256=sha(payload/'bundle.json'),logical_bytes=sum(v['bytes'] for v in bundle['files'].values()))
    (base/'preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))

if __name__=='__main__':main()
