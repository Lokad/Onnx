"""Reuse the qualified replay binary and immutable native model references."""
from common import *
import shutil
import subprocess


def main():
    assert not BASE.exists()
    assert pin(PROOF/'closed.json')['sha256']=='9ce5e59054ad6f6e1b5c1e12c8568c77558eafadc8850a5a3a16b36505c4ff27'
    proof=read(PROOF/'closed.json');assert proof['qualified'] and all(absent(i) for i in proof['identities'])
    for name,expected in proof['files'].items():assert pin(ROOT/name)==expected,name
    assert pin(OLD/'closed.json')['sha256']=='b2631477f8f4b227ececf50ea48fed12f68581d706d56d8dc3c4f268819718a7'
    closure=read(OLD/'closed.json');meta=read(OLD/'payload/frozen.json')
    assert pin(OLD/'payload/frozen.json')==closure['files']['payload/frozen.json']
    previous=read(PROOF/'manifest.json');BASE.mkdir();files={}
    def bind(path):files[rel(path)]=pin(path)
    bind(PROOF/'closed.json');bind(PROOF/'manifest.json');bind(OLD/'closed.json');bind(OLD/'payload/frozen.json')
    replay=OLD/'payload'/meta['paths']['replay'];cores={}
    for role in ('baseline','candidate'):
        destination=BASE/'runtimes'/role;destination.mkdir(parents=True)
        for path in replay.iterdir():
            if not path.is_file():continue
            key=path.relative_to(OLD/'payload').as_posix()
            assert pin(path)==meta['files'][key]==closure['files']['payload/'+key]
            bind(path);shutil.copyfile(path,destination/path.name)
        for suffix in ('dll','pdb'):
            path=PROOF/'runtimes'/role/f'Lokad.Onnx.{suffix}'
            assert pin(path)==proof['files'][rel(path)];bind(path)
            shutil.copyfile(path,destination/path.name)
        cores[role]=pin(destination/'Lokad.Onnx.dll');assert cores[role]==previous['cores'][role]
        for path in destination.iterdir():
            if path.is_file():bind(path)
    runner=pin(BASE/'runtimes/baseline/Replay.dll')
    assert runner==pin(BASE/'runtimes/candidate/Replay.dll')==meta['files'][meta['paths']['replay']+'/Replay.dll']
    shared=read(REFERENCE/'manifest.json')
    assert pin(REFERENCE/'manifest.json')==meta['assets'][rel(REFERENCE/'manifest.json')];bind(REFERENCE/'manifest.json')
    for model in shared['models']:
        for asset in model['assets']:
            path=ROOT/asset['file'];assert pin(path)=={k:asset[k] for k in ('bytes','sha256')}==meta['assets'][rel(path)];bind(path)
        for scenario in model['scenarios']:
            for step in scenario['steps']:
                for record in step['inputs']+step['outputs']:
                    path=REFERENCE/record['file'];assert pin(path)['sha256']==record['sha256']
                    assert pin(path)==meta['assets'][rel(path)];bind(path)
    for name in CASES:
        fixture=E5/(name+'.json');f=read(fixture);assert f['name']==name
        assert pin(fixture)==previous['files'][rel(fixture)];bind(fixture)
        path=E5/f['reference_file'];assert pin(path)['sha256']==f['reference_sha256'];bind(path)
    bind(ROOT/'models/multilingual-e5-small/model.onnx')
    for name in ('Program.cs','Replay.csproj'):
        path=OLD/'payload/tests/e5/fingerprint-product'/name
        assert pin(path)==meta['files'][path.relative_to(OLD/'payload').as_posix()];bind(path)
    for path in Path(__file__).parent.iterdir():
        if path.is_file():bind(path)
    write(BASE/'manifest.json',dict(protocol='profiler-shared-models-v1',source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
          jobs=JOBS,limits=LIMITS,cores=cores,runner=runner,references=dict(e5=rel(E5),shared=rel(REFERENCE)),files=files))
    print(json.dumps(dict(manifest=pin(BASE/'manifest.json'),jobs=len(JOBS),runner=runner,files=len(files))))


if __name__=='__main__':main()
