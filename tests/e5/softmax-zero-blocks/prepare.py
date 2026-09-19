"""Freeze a small AMD payload after local checks, without any model weights."""
from pathlib import Path
import argparse,hashlib,json,subprocess,tarfile

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();root=Path(__file__).resolve().parents[3];base=a.artifact.resolve()
    check=json.loads((base/'local-check.json').read_text(encoding='utf-8'));assert check['checkOnly'] and check['tensorCases']==918 and check['refusals']==4
    payload=base/'payload';payload.mkdir()
    sources=[('run.py',Path(__file__).with_name('run.py')),('campaign_processes.py',root/'eng/campaign_processes.py'),('prospective-plan.md',root/'.agent/m2-softmax-zero-blocks-20260919.md'),
        ('source/Program.cs',Path(__file__).with_name('Program.cs')),('source/generate.py',Path(__file__).with_name('generate.py')),('source/Probe.csproj',Path(__file__).with_name('Probe.csproj'))]
    sources.extend(('kernels/'+p.name,p) for p in (base/'kernels').iterdir() if p.is_file())
    sources.extend(('bin/'+p.name,p) for p in (base/'bin').iterdir() if p.suffix in ('.dll','.json','.pdb'))
    files={}
    for rel,path in sources:
        target=payload/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());files[rel]=dict(bytes=target.stat().st_size,sha256=sha(target))
    assert check['probe_sha256']==files['bin/Probe.dll']['sha256'] and check['core_sha256']==files['bin/Lokad.Onnx.dll']['sha256']
    manifest=dict(schema=1,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),local_check_sha256=sha(base/'local-check.json'),files=files,
        protocol=dict(workers=8,orders=list(range(8)),codegen_workers=1,shapes=['8','30','pad128','128','512','pad512'],modes=['actual','copied','skip','adaptive'],samples=9,iterations=[128,128,16,16,2,2],conditioning_seconds_per_shape_mode=1,normal_tiering=True,target_cpu=2))
    (payload/'bundle.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    archive=base/'bundle.tar.gz'
    with tarfile.open(archive,'x:gz') as t:
        for path in sorted(payload.rglob('*')):
            if path.is_file():t.add(path,arcname=path.relative_to(payload).as_posix())
    print('bundle',archive.stat().st_size,sha(archive))

if __name__=='__main__':main()
