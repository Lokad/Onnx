"""Retire a closed private cache and byte-identically retained diagnostic outputs."""
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/parakeet-rational-model-headroom-20260927'


def pin(path):
    data = path.read_bytes()
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    assert not OUT.exists(), 'One-time cache retirement already attempted'
    sources = [
        ('parakeet-rational-sigmoid-build-amd-20260927', 'lokad-parakeet-rational-sigmoid-build-20260927',
         '3301ae58b42e1fd51435f54191cf4bba42c6cbf8e2ad6bf4279af67ae5d3d89a'),
    ]
    specs = []
    for local, remote, digest in sources:
        base = ROOT/'artifacts'/local
        assert pin(base/'closed.json')['sha256'] == digest
        proof = json.loads((base/'closed.json').read_text())
        assert proof['passed'] and proof['analysis'] == pin(base/'analysis.json')
        for kind in ['build', 'capture']:
            path = base/(kind+'-collected')/(kind+'-collection.json')
            assert pin(path) == proof['files'][path.relative_to(base).as_posix()]
        specs.append(dict(base='/dev/shm/'+remote,
            collections={k:pin(base/(k+'-collected')/(k+'-collection.json')) for k in ['build','capture']},
            built=pin(base/'build-collected/built.json'), spec=pin(base/'bundle/spec.json')))
    outputs = []
    for local, remote, digest, names in [
        (sources[0][0], sources[0][1], sources[0][2], ['instructions.json']),
        ('parakeet-rational-sigmoid-fallback-diagnostic-amd-20260927',
         'lokad-parakeet-rational-sigmoid-fallback-diagnostic-20260927',
         'c6f0a34eb3467c69829ad958194f5f88375b83abb12fad8450c233bc41586049',
         [n+ext for n in ['current-0','candidate-1','candidate-2','current-3'] for ext in ['.json','.stdout']]),
    ]:
        base = ROOT/'artifacts'/local
        assert pin(base/'closed.json')['sha256'] == digest
        proof = json.loads((base/'closed.json').read_text())
        copies = {}
        for name in names:
            path = base/'capture-collected/logs'/name
            assert pin(path) == proof['files'][path.relative_to(base).as_posix()]
            copies['logs/'+name] = pin(path)
        receipt = base/'capture-collected/capture-collection.json'
        assert pin(receipt) == proof['files'][receipt.relative_to(base).as_posix()]
        outputs.append(dict(base='/dev/shm/'+remote,files=copies,collection=pin(receipt)))
    tool = ROOT/'tests/parakeet/rational-sigmoid-build/run.py'
    loader = importlib.util.spec_from_file_location('headroom_transport', tool)
    module = importlib.util.module_from_spec(loader); loader.loader.exec_module(module)
    OUT.mkdir()
    (OUT/'intent.json').write_text(json.dumps(dict(specs=specs,outputs=outputs,tool=pin(Path(__file__))), indent=2))
    script = '''
from pathlib import Path
import hashlib,json,signal,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
signal.alarm(60)
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def read(p):return json.loads(p.read_text())
def live(i):
    try:
        p=psutil.Process(i['pid']);return p.create_time()==i['birth'] and p.status()!=psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:return False
own=psutil.Process();parents={own.pid,*[p.pid for p in own.parents()]}
assert psutil.boot_time()==1789634288.0
for p in psutil.process_iter(['name','cmdline']):
    if p.pid in parents:continue
    assert p.info['name'] not in ['dotnet','perf']
    assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
feed=Path('/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed').resolve()
archives={p.name.lower():p for p in feed.glob('*.nupkg')}
checked=[];retained=[];protected={};before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for spec in SPECS:
    base=Path(spec['base']).resolve();target=base/'packages'
    assert base.parent==Path('/dev/shm') and target.resolve()==target and target.is_dir() and not target.is_symlink()
    assert pin(base/'spec.json')==spec['spec'] and pin(base/'built.json')==spec['built']
    for kind,wanted in spec['collections'].items():
        path=base/(kind+'-collection.json');assert pin(path)==wanted
        receipt=read(path);assert receipt['terminal'] and receipt['code']==0 and not any(live(i) for i in receipt['identities'])
    frozen=read(base/'spec.json')
    for name,wanted in frozen['files'].items():
        path=(base/name).resolve();assert not path.is_relative_to(target)
        assert pin(path)==wanted;protected[str(path)]=wanted
    for name,wanted in frozen['external'].items():
        path=Path(name).resolve();assert not path.is_relative_to(target)
        assert pin(path)==wanted;protected[str(path)]=wanted
    for name,wanted in read(base/'built.json')['runtime'].items():
        path=base/'runtime'/name;assert pin(path)==wanted;protected[str(path)]=wanted
    packages=list(target.glob('*/*'));assert packages
    for package in packages:
        assert package.is_dir() and not package.is_symlink()
        package_archives=list(package.glob('*.nupkg'));assert len(package_archives)==1
        archive=package_archives[0];canonical=archives[archive.name.lower()]
        assert pin(archive)==pin(canonical)
        retained.append(dict(private=str(archive),canonical=str(canonical),identity=pin(canonical)))
    for path in target.rglob('*'):
        assert not path.is_symlink() and path.resolve().is_relative_to(target)
        if path.is_file():
            st=path.stat();assert st.st_nlink==1
            checked.append(dict(path=str(path),identity=pin(path),physical_bytes=st.st_blocks*512))
assert len(retained)==23
for output in OUTPUTS:
    base=Path(output['base']).resolve();assert base.parent==Path('/dev/shm')
    assert pin(base/'capture-collection.json')==output['collection']
    receipt=read(base/'capture-collection.json')
    assert receipt['terminal'] and receipt['code']==0 and not any(live(i) for i in receipt['identities'])
    spec=read(base/'spec.json')
    for name,wanted in spec['files'].items():protected[str((base/name).resolve())]=wanted
    for name,wanted in spec['external'].items():protected[str(Path(name).resolve())]=wanted
    for name,wanted in output['files'].items():
        path=base/name;assert path.resolve().is_relative_to(base/'logs') and not path.is_symlink()
        assert str(path) not in protected and pin(path)==wanted==receipt['files'][name]
        st=path.stat();assert st.st_nlink==1
        checked.append(dict(path=str(path),identity=wanted,physical_bytes=st.st_blocks*512,retained_locally=True))
assert len({r['path'] for r in checked})==len(checked)
for row in checked:
    path=Path(row['path']);assert pin(path)==row['identity'];path.unlink()
for spec in SPECS:
    target=Path(spec['base'])/'packages'
    for path in sorted((p for p in target.rglob('*') if p.is_dir()),key=lambda p:len(p.parts),reverse=True):path.rmdir()
    target.rmdir()
for name,wanted in protected.items():assert pin(Path(name))==wanted
for row in retained:assert pin(Path(row['canonical']))==row['identity']
print(json.dumps(dict(passed=True,files=checked,retained_archives=retained,protected_files=len(protected),
    physical_bytes_freed=sum(r['physical_bytes'] for r in checked),before=before,
    after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
'''.replace('SPECS', repr(specs)).replace('OUTPUTS', repr(outputs))
    (OUT/'remote.py').write_text(script, encoding='utf8')
    result = module.ssh(script)
    (OUT/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    (OUT/'closed.json').write_text(json.dumps(dict(passed=result['passed'],result=pin(OUT/'result.json'),
        script=pin(OUT/'remote.py'),intent=pin(OUT/'intent.json')), indent=2)+'\n', encoding='utf8')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','retained_archives']}))


if __name__ == '__main__':main()
