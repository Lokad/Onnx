"""Freeze proof first; timing requires its closed, passed machine-code evidence."""
from pathlib import Path
import argparse,json,hashlib,shutil,subprocess,tarfile

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def pin(p):return dict(bytes=p.stat().st_size,sha256=sha(p))

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--phase',choices=['proof','timing'],required=True)
    p.add_argument('--proof-artifact',type=Path);a=p.parse_args()
    base=a.artifact.resolve();root=Path(__file__).resolve().parents[3];lane=Path(__file__).parent
    build=base if a.phase=='proof' else a.proof_artifact.resolve()
    payload=base/'payload';assert not payload.exists() and not (base/'payload.tar.gz').exists()
    source=json.loads((build/'source/source.json').read_text())
    for name,value in source['sources'].items():assert pin(root/name)==value,name
    for name,value in source['files'].items():assert pin(build/'source'/name)==value,name
    for name in ['Program.cs','Probe.csproj']:assert pin(lane/name)==source['files'][name]
    assert sha(build/'bin/Lokad.Onnx.dll')=='05884cfd524cc7130321f5dc1bcd0af17dddc7b97e8428d2d2f59e00edb795c2'
    old=root/'artifacts/projection-input-pack-20260919';old_receipt=json.loads((old/'closed.json').read_text())
    assert sha(old/'closed.json')=='0075d7770b1a1b8c3647db20a17ac09263d23fff5bfba7d60ffc2e2d6c5aa54f'
    for name,value in old_receipt['files'].items():assert pin(old/name)==value,name
    for name in ['Original.cs','InputPacked.cs']:assert pin(build/'source'/name)==pin(old/'payload/source'/name)
    files={}
    for f in (build/'source').iterdir():
        if f.is_file():files['source/'+f.name]=f
    for f in (build/'bin').iterdir():
        if f.suffix in ('.dll','.json'):files['bin/'+f.name]=f
    for f in lane.glob('*.py'):files[f.name]=f
    files['process_support.py']=root/'tests/parakeet/recording-amd/remote.py'
    files['campaign_processes.py']=root/'eng/campaign_processes.py'
    files['reference/previous-generate.py']=root/'tests/e5/projection-input-pack/generate.py'
    files['reference/previous-closed.json']=old/'closed.json'
    files['plan.md']=root/'.agent/m2-projection-input-pointer-20260919.md'
    if a.phase=='timing':
        closed=json.loads((build/'closed.json').read_text());proof=json.loads((build/'audit.json').read_text())
        assert closed['proof_pass'] and proof['proof_pass'] and closed['audit_sha256']==sha(build/'audit.json')
        for name,value in closed['files'].items():assert pin(build/name)==value,name
        assert {name:pin(build/'bin'/name) for name in closed['binary_files']}==closed['binary_files']
        files['reference/proof-closed.json']=build/'closed.json';files['reference/proof-audit.json']=build/'audit.json'
    else:
        host=json.loads(subprocess.check_output(['dotnet',str(build/'bin/Probe.dll'),'--host'],text=True))
        assert host==dict(Avx512F=False,Fma=True,unsupported_refusal_checked=True,packing_checks=54)
    payload.mkdir(parents=True)
    for name,path in files.items():
        target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    bundle=dict(schema=1,phase=a.phase,source_commit=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),
        files={name:pin(path) for name,path in files.items()},local_sources={name:path.relative_to(root).as_posix() for name,path in files.items()})
    (payload/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n',encoding='utf-8')
    archive=base/'payload.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
        for path in sorted(payload.rglob('*')):
            if path.is_file():tar.add(path,arcname=path.relative_to(payload).as_posix(),recursive=False)
    record=dict(**pin(archive),bundle_sha256=sha(payload/'bundle.json'),logical_bytes=sum(v['bytes'] for v in bundle['files'].values()))
    (base/'preparation.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))

if __name__=='__main__':main()
