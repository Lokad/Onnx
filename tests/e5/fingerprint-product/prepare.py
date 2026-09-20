"""Archive committed qualification sources, build them, and freeze a bounded replay."""
from pathlib import Path
import argparse,hashlib,json,shutil,subprocess,tarfile

ROOT=Path(__file__).resolve().parents[3]

def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())

def write(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True);args=parser.parse_args()
    base=args.artifact.resolve();base.mkdir(exist_ok=True)
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit sources before freezing'
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    source=base/'source';assert not source.exists();source.mkdir()
    roots=['src','tests/Lokad.Onnx.Backend.Tests','tests/Lokad.Onnx.Tensors.Tests','tests/Shared','tests/e5/fingerprint-product',
           'global.json','Lokad.Onnx.slnx','README.md','LICENSE.txt','CHANGELOG.md','icon.png']
    # Include every first-party C# file for the existing source-contract tests.
    tracked=subprocess.check_output(['git','ls-files','tests'],cwd=ROOT,text=True).splitlines()
    roots += [p for p in tracked if p.endswith('.cs') and not any(p.startswith(r+'/') for r in roots)]
    archive=base/'source.tar';assert not archive.exists()
    subprocess.run(['git','archive','--format=tar','--output='+str(archive),revision,*roots],cwd=ROOT,check=True)
    with tarfile.open(archive) as tar:tar.extractall(source,filter='data')
    original={p.relative_to(source).as_posix():pin(p) for p in sorted(source.rglob('*')) if p.is_file()}
    write(base/'source-identity.json',dict(revision=revision,archive=pin(archive),files=original))
    for label,project in [('cli','src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
                          ('backend','tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
                          ('tensors','tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'),
                          ('replay','tests/e5/fingerprint-product/Replay.csproj')]:
        command=['dotnet','build',project,'-c','Release','--tl:off','--nologo','-v','minimal',
                 '-p:SourceRevisionId='+revision,'-p:RepositoryCommit='+revision]
        with (base/('build-'+label+'.log')).open('x') as log:
            subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
    for name,wanted in original.items():assert pin(source/name)==wanted,name
    payload=base/'payload';assert not payload.exists();payload.mkdir()
    for name in original:
        target=payload/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
    paths={'cli':'src/Lokad.Onnx.CLI/bin/Release/net10.0',
           'backend':'tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0','tensors':'tests/Lokad.Onnx.Tensors.Tests/bin/Release/net10.0',
           'replay':'tests/e5/fingerprint-product/bin/Release/net10.0'}
    for path in paths.values():shutil.copytree(source/path,payload/path)
    cores=[pin(payload/path/'Lokad.Onnx.dll') for path in paths.values()];assert all(c==cores[0] for c in cores)
    shutil.copytree(ROOT/'artifacts/e5-paired-aa-v2-20260920/payload/inputs',payload/'inputs')
    (payload/'qualification').mkdir()
    for name in ['run.py','audit.py','collect.py']:
        shutil.copyfile(source/'tests/e5/fingerprint-product'/name,payload/'qualification'/name)
    shutil.copyfile(ROOT/'.agent/m2-fingerprint-product-20260920.md',payload/'qualification/prospective-plan.md')
    assets={};reference=ROOT/'artifacts/shared-regression-20260918/reference'
    manifest=json.loads((reference/'manifest.json').read_text())
    for model in manifest['models']:
        for asset in model['assets']:
            assert pin(ROOT/asset['file'])=={k:asset[k] for k in ['bytes','sha256']}
            assets[asset['file']]=pin(ROOT/asset['file'])
        for scenario in model['scenarios']:
            for step in scenario['steps']:
                for item in step['inputs']+step['outputs']:
                    path=reference/item['file'];assert pin(path)['sha256']==item['sha256']
                    assets[path.relative_to(ROOT).as_posix()]=pin(path)
    assets['artifacts/shared-regression-20260918/reference/manifest.json']=pin(reference/'manifest.json')
    assets['models/multilingual-e5-small/model.onnx']=pin(ROOT/'models/multilingual-e5-small/model.onnx')
    jobs=[dict(name=kind+'-'+str(enabled),kind=kind,enabled=enabled) for enabled in [0,1] for kind in ['backend','tensors','e5','shared']]
    meta=dict(schema=1,source_revision=revision,source_archive=pin(archive),core=cores[0],paths=paths,assets=assets,jobs=jobs,
              limits=dict(seconds=600,rss=12*1024**3,available=1024**3),
              files={p.relative_to(payload).as_posix():pin(p) for p in sorted(payload.rglob('*')) if p.is_file()})
    write(payload/'frozen.json',meta)
    package=base/'payload.tar.gz'
    with tarfile.open(package,'x:gz') as tar:
        for name in list(meta['files'])+['frozen.json']:tar.add(payload/name,arcname=name,recursive=False)
    write(base/'preparation.json',dict(archive=pin(package),frozen=pin(payload/'frozen.json'),source=pin(base/'source-identity.json')))
    print(json.dumps(dict(source=revision,core=cores[0],files=len(meta['files']),archive=pin(package))))

if __name__=='__main__':main()
