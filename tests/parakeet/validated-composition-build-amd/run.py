"""Build the exact admitted composition, then qualify tensor behavior in both modes."""
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
loader=importlib.util.spec_from_file_location('slice_build_transport',TOOLS.parent/'slice-materialization-build-amd/run.py')
slice_build=importlib.util.module_from_spec(loader);loader.loader.exec_module(slice_build)
transport=slice_build.transport
BASE=ROOT/'artifacts/parakeet-validated-composition-build-amd-20260924'
REMOTE='/dev/shm/lokad-parakeet-validated-composition-build-20260924'
SOURCE=ROOT/'artifacts/parakeet-validated-composition-source-20260924'
PRIOR=ROOT/'artifacts/parakeet-prepared-recurrence-build-amd-20260924'
REMOTE_PRIOR='/dev/shm/lokad-parakeet-prepared-recurrence-build-20260924/runtime'
SLICE_BUILD=ROOT/'artifacts/parakeet-slice-materialization-build-amd-v2-20260924'
pin,read,write,ssh=slice_build.pin,slice_build.read,slice_build.write,slice_build.ssh
PRELUDE=slice_build.PRELUDE.replace(slice_build.REMOTE,REMOTE)
transport.BASE,transport.REMOTE,transport.PRELUDE=BASE,REMOTE,PRELUDE


def prepare():
    from prepare_source import inspect, APPS
    inspect();source=read(SOURCE/'prepared.json');assert source['passed'] and not BASE.exists()
    for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
    for label,folder in APPS.items():
        assert pin(folder/'closed.json')==source['admissions'][label]['closure']
        assert read(folder/'closed.json')['admitted']
    BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir()
    def put(name,data):
        path=bundle/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data if isinstance(data,bytes) else data.encode())
    for name in source['source']:put('source/'+name,(SOURCE/'source'/name).read_bytes())
    put('source/tests/Lokad.Onnx.Tensors.Tests/SliceCandidateIdentityTests.cs',
        (TOOLS.parent/'slice-materialization-build-amd/IdentityTests.cs.txt').read_bytes())
    bridge=(TOOLS.parent/'selected-profile-build-amd/Bridge.cs.txt').read_text(encoding='utf8')
    assert bridge.count('new[] { "SampledAudio.dll" }')==1
    put('bridge-source/Program.cs',bridge.replace('new[] { "SampledAudio.dll" }','new[] { "Lokad.Onnx.dll" }'))
    put('bridge-source/Bridge.csproj',(TOOLS.parent/'selected-profile-build-amd/Bridge.csproj').read_bytes())
    put('bridge-source/global.json',(ROOT/'global.json').read_bytes())
    put('common.py',(TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('candidate_build.py',(TOOLS.parent/'slice-materialization-build-amd/remote.py').read_bytes())
    for name in ['remote.py','README.md']:put(name,(TOOLS/name).read_bytes())
    methods_path=SLICE_BUILD/'build-collected/inventory/instructions.json'
    assert read(SLICE_BUILD/'build-review.json')['inventory']==pin(methods_path)
    row,=read(methods_path)['observations']
    assert row['after_sha256']=='bafdb0069c4809251fd4e287ad094076ed5e01cefe8be7be6dc6446f8887a9da'
    assert len(row['differences'])==len(row['added'])==1
    put('copy-methods.json',json.dumps(dict(source_inventory=pin(methods_path),methods=row['candidate_methods']),indent=2))
    current=PRIOR/'collected/runtime'
    core=pin(current/'Lokad.Onnx.dll');data=pin(current/'Lokad.Onnx.Data.dll')
    assert dict(**{'Lokad.Onnx.dll':core,'Lokad.Onnx.Data.dll':data})==source['admissions']['recurrence']['identities']['candidate']
    spec=dict(boot=1789634288.0,prior=REMOTE_PRIOR,
        external={REMOTE_PRIOR+'/'+p.name:pin(p) for p in current.iterdir() if p.is_file()},
        source_prepared=pin(SOURCE/'prepared.json'),admissions=source['admissions'],core=core,data=data,
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=300),
        minimum_free=1024**3,output_limit=512*1024**2,expected_tests=369,expected_skipped=0,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json',spec)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as archive:
        for path in bundle.rglob('*'):
            if path.is_file():archive.add(path,arcname=path.relative_to(bundle).as_posix(),recursive=False)
    write(BASE/'prepared.json',dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},transport=pin(TOOLS.parent/'managed-phase-amd/run.py')))
    print(json.dumps(dict(prepared=True,archive=pin(BASE/'payload.tar.gz'),source_files=425)))


def observe(kind):
    slice_build.BASE,slice_build.PRELUDE=BASE,PRELUDE
    slice_build.observe(kind)


if __name__=='__main__':
    action=sys.argv[1]
    if action=='prepare':prepare()
    elif action=='stage':transport.stage()
    elif action=='observe':observe(sys.argv[2])
    else:dict(launch=transport.launch,collect=transport.collect)[action](sys.argv[2])
