"""Stage immutable closed-proof inputs, then freeze the new host and launch tools."""
from pathlib import Path
import argparse,json,shutil,subprocess,tarfile
from common import CORE,ORIGIN,LIMITS,pin,read,write

ROOT=Path(__file__).resolve().parents[3]

def stage(base):
    old=ROOT/'artifacts/e5-layernorm-output-20260920';assert pin(old/'closed.json')['sha256']==ORIGIN
    closed=read(old/'closed.json');assert closed['passed'] is True
    payload=base/'payload';payload.mkdir(parents=True);selected={}
    def copy(relative,target):
        source=old/relative;assert pin(source)==closed['files'][relative] and not source.is_symlink()
        destination=payload/target;destination.parent.mkdir(parents=True,exist_ok=True);assert not destination.exists()
        shutil.copyfile(source,destination);assert pin(destination)==closed['files'][relative];selected[target]=relative
    for source in sorted((old/'capture').rglob('*')):
        if source.is_file():copy(source.relative_to(old).as_posix(),'capture/'+source.relative_to(old/'capture').as_posix())
    assert len([n for n in selected if n.startswith('capture/')])==511
    for source,target in [('closed-source/Proof.cs','source/Proof.cs'),('generated/Kernels.cs','source/Kernels.cs'),
                          ('generated/source.json','source/generated.json'),('closed-source/audit.py','tools/original_audit.py'),
                          ('audit.json','origin/local-audit.json'),('nan-fixture-scope.json','origin/nan-fixture-scope.json')]:copy(source,target)
    for name in ['Lokad.Onnx.dll','Google.Protobuf.dll']:copy('bin-final/'+name,'product/'+name)
    assert pin(ROOT/'tests/e5/layernorm-output/Proof.cs')==pin(payload/'source/Proof.cs')
    assert pin(ROOT/'src/Lokad.Onnx/TensorOps.Norm.cs')['sha256']==read(payload/'source/generated.json')['source_sha256']
    shutil.copyfile(ROOT/'src/Lokad.Onnx/TensorOps.Norm.cs',payload/'source/TensorOps.Norm.cs')
    (payload/'origin').mkdir(exist_ok=True);shutil.copyfile(old/'closed.json',payload/'origin/closed.json')
    write(payload/'origin/selected.json',selected)
    write(base/'build-inputs.json',{name:pin(Path(__file__).parent/name) for name in ['Program.cs','Probe.csproj']})
    print('Staged',len(selected),'closed-receipt-bound files; no inference performed.')

def freeze(base):
    payload=base/'payload';assert not (payload/'bundle.json').exists()
    for name,want in read(base/'build-inputs.json').items():assert pin(Path(__file__).parent/name)==want
    assert pin(payload/'bin/Lokad.Onnx.dll')['sha256']==CORE
    local=read(base/'local-identity.json');assert local['passed'] and local['identity']['mode']=='identity'
    assert local['identity']['probe_sha256']==pin(payload/'bin/LayerNormAmdProof.dll')['sha256']
    assert 'OK' in (base/'unit-tests.log').read_text() and '0 Warning(s)' in (base/'build.log').read_text()
    for source in Path(__file__).parent.iterdir():
        if source.is_file():
            destination=payload/'tools'/source.name;assert not destination.exists();shutil.copyfile(source,destination)
    files={p.relative_to(payload).as_posix():pin(p) for p in sorted(payload.rglob('*')) if p.is_file()}
    write(payload/'bundle.json',dict(schema=1,protocol='layernorm-amd-arithmetic-code-v1',source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        core_sha256=CORE,origin_sha256=ORIGIN,limits=LIMITS,phases=['proof','code'],cases=915,captures=125,comparisons=32182096,files=files))
    with tarfile.open(base/'payload.tar.gz','x:gz') as tar:
        for p in sorted(payload.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(payload).as_posix(),recursive=False)
    write(base/'frozen.json',dict(bundle=pin(payload/'bundle.json'),archive=pin(base/'payload.tar.gz'),local_identity=pin(base/'local-identity.json'),files=len(files)))
    print('Frozen',len(files),'files;',pin(base/'payload.tar.gz'))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['stage','freeze']);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args()
    globals()[args.action](args.artifact.resolve())
