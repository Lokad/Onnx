"""Freeze one diagnostic on the isolated candidate; do not promote the release."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from isolated_baseline import qualify, source_path, RELEASE

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-feed-forward-cost-amd-20260925'
REMOTE = '/dev/shm/lokad-parakeet-feed-forward-cost-20260925'
CORE_SOURCE = ROOT / 'artifacts/parakeet-feed-forward-cost-source-20260925'
DATA_SOURCE = ROOT / 'artifacts/parakeet-feed-forward-cost-observer-source-20260925'
MODELS = ROOT / 'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
REMOTE_MODELS = '/dev/shm/lokad-parakeet-slice-dense-conversion-models-20260925'
OBSERVER = ROOT / 'artifacts/parakeet-slice-materialization-profile-amd-20260924'
REMOTE_OBSERVER = '/dev/shm/lokad-parakeet-slice-materialization-profile-20260924'
APP = ROOT / 'artifacts/parakeet-observed-dense-where-app-amd-20260924'
REMOTE_APP = '/dev/shm/lokad-parakeet-observed-dense-where-app-20260924'
MANIFEST = 'manifests/candidate-parakeet.json'
BRIDGE = ROOT / 'artifacts/parakeet-observed-dense-where-root-amd-v2-20260924/bundle/bridge'
COMMON = TOOLS.parent / 'managed-phase-amd/remote.py'
loader = importlib.util.spec_from_file_location('cost_transport', TOOLS.parent / 'managed-phase-amd/run.py')
transport = importlib.util.module_from_spec(loader)
loader.loader.exec_module(transport)
pin, read, write, ssh, SSH = transport.pin, transport.read, transport.write, transport.ssh, transport.SSH
PRELUDE = transport.PRELUDE.replace(transport.REMOTE, REMOTE)
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE


def prerequisites():
    isolated = qualify()
    candidate = dict(source_files=isolated['source_files'])
    measured = isolated['product']
    assert measured['Lokad.Onnx.dll']['sha256'] == '49c3a958850d3e57daa2b7e29e6bd15ff9fc4f27af098d8ce44d8f20b9e065e8'
    assert measured['Lokad.Onnx.Data.dll']['sha256'] == 'a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
    core = read(CORE_SOURCE / 'review.json')
    observer = read(DATA_SOURCE / 'review.json')
    assert pin(CORE_SOURCE / 'review.json')['sha256'] == 'be4e3a92e3a4c1673553f474e85bb5fc938b7757e68bc30cd8c29108cd80d865'
    assert pin(DATA_SOURCE / 'review.json')['sha256'] == '61b75a571751e6c29395747eea91169b69f76867b477ac08c82918759a752204'
    assert core['passed'] and observer['passed'] and observer['exact_original_byte_restoration']
    assert core['generator'] == pin(TOOLS / 'source.py') and observer['reviewer'] == pin(TOOLS / 'verify_source.py')
    for name, row in core['files'].items():
        assert pin(ROOT / 'src/Lokad.Onnx' / name) == row['original']
        assert pin(CORE_SOURCE / name) == row['diagnostic']
    assert pin(CORE_SOURCE / 'FeedForwardCostStages.cs') == core['added_source']
    assert pin(ROOT / 'src/Lokad.Onnx.Data/ParakeetTranscriber.cs') == observer['original_transcriber']
    assert pin(DATA_SOURCE / 'ParakeetTranscriber.cs') == observer['diagnostic_transcriber']
    assert pin(DATA_SOURCE / 'PhaseProbe.cs') == pin(TOOLS / 'PhaseProbe.cs.txt') == observer['observer']
    assert pin(OBSERVER / 'build-collected/runtime-observed/SampledAudio.dll') == observer['consumer']
    return candidate, measured, core, observer, isolated


def prepare():
    assert all((TOOLS / name).is_file() for name in ['audit.py', 'analyze.py']), 'Finish the capture audit before freezing tools'
    candidate, product, core, observer, isolated = prerequisites()
    from reference import references
    reference = references()
    assert not BASE.exists(), 'Preserve the prepared diagnostic'
    # Check every original input before creating a new artifact directory.
    manifest = read(APP / 'collected' / MANIFEST)
    assert len(manifest['cases']) == 20 and manifest['warmup_passes'] == 1 and manifest['measured_passes'] == 3
    BASE.mkdir()
    bundle = BASE / 'bundle'
    bundle.mkdir()

    def put(name, content):
        path = bundle / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(content if isinstance(content, bytes) else content.encode())

    for name in candidate['source_files']:
        source = source_path(name)
        assert pin(source) == candidate['source_files'][name]
        if name.startswith('src/Lokad.Onnx/') and Path(name).name in core['files']:
            source = CORE_SOURCE / Path(name).name
        put('core-source/' + name, source.read_bytes())
    put('core-source/src/Lokad.Onnx/FeedForwardCostStages.cs', (CORE_SOURCE / 'FeedForwardCostStages.cs').read_bytes())
    for path in sorted((ROOT / 'src/Lokad.Onnx.Data').glob('*.cs')):
        source = DATA_SOURCE / path.name if path.name == 'ParakeetTranscriber.cs' else path
        assert pin(path) == candidate['source_files'][path.relative_to(ROOT).as_posix()]
        put('data-source/' + path.name, source.read_bytes())
    put('data-source/PhaseProbe.cs', (DATA_SOURCE / 'PhaseProbe.cs').read_bytes())
    refs = ['Lokad.Onnx', 'Google.Protobuf', 'FastBertTokenizer', 'Lokad.Tokenizers', 'SixLabors.ImageSharp']
    references = ''.join(f'<Reference Include="{name}"><HintPath>$(FrozenProductDirectory)/{name}.dll</HintPath></Reference>' for name in refs)
    put('data-source/ObserverData.csproj', '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><LangVersion>11.0</LangVersion><AssemblyName>Lokad.Onnx.Data</AssemblyName></PropertyGroup><ItemGroup>' + references + '</ItemGroup></Project>')
    put('data-source/global.json', (ROOT / 'global.json').read_bytes())
    for name in ['Bridge.dll', 'Bridge.deps.json', 'Bridge.runtimeconfig.json']:
        assert pin(RELEASE / 'bundle/bridge' / name) == pin(BRIDGE / name)
        put('bridge/' + name, (BRIDGE / name).read_bytes())
    put('common.py', COMMON.read_bytes())
    put('remote.py', (TOOLS / 'vm.py').read_bytes())
    for name in ['README.md', 'il_check.py']:
        put(name, (TOOLS / name).read_bytes())
    for name, path in [('core-source-review.json', CORE_SOURCE / 'review.json'),
                       ('observer-source-review.json', DATA_SOURCE / 'review.json'),
                       ('selected-release-closure.json', RELEASE / 'closed.json'),
                       ('selected-release-analysis.json', RELEASE / 'analysis.json')]:
        put('evidence/' + name, path.read_bytes())
    put('evidence/isolated-baseline.json', json.dumps({k:v for k,v in isolated.items() if k != 'inventory'}, separators=(',', ':'), allow_nan=False))
    put('evidence/cost-reference.json', json.dumps(reference, separators=(',', ':'), allow_nan=False))
    runtime_files = {}
    external = {}
    for path in (OBSERVER / 'build-collected/runtime-observed').iterdir():
        if not path.is_file():
            continue
        if path.name in product:
            source = MODELS / 'collected/runtimes/candidate' / path.name
            remote_path = REMOTE_MODELS + '/runtimes/candidate/' + path.name
        else:
            source = path
            remote_path = REMOTE_OBSERVER + '/runtime-observed/' + path.name
        runtime_files[path.name] = pin(source)
        external[remote_path] = pin(source)
    for row in manifest['models'].values():
        external[row['path']] = {key: row[key] for key in ['bytes', 'sha256']}
    for row in [manifest['reference'], *[case['pcm'] for case in manifest['cases']]]:
        external[REMOTE_APP + '/assets/' + row['path']] = {key: row[key] for key in ['bytes', 'sha256']}
    for name in [MANIFEST, 'runtime/protocol.py', 'runtime/campaign_processes.py']:
        external[REMOTE_APP + '/' + name] = pin(APP / 'collected' / name)
    spec = dict(boot=1789634288.0, external=external, product=product, consumer=observer['consumer'],
        diagnostic_references=reference['inputs'],
        consumer_runtime=REMOTE_OBSERVER + '/runtime-observed',
        product_runtime=REMOTE_MODELS + '/runtimes/candidate', original_runtime_files=runtime_files,
        selected_release_closure=pin(RELEASE / 'closed.json'), candidate_source=candidate['source_files'],
        isolated_evidence=isolated['evidence'], release_admitted=False, diagnostic_only=True,
        app=REMOTE_APP, manifest=MANIFEST,
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3, tmpfs_before=1024**3, rss=3*1024**3, seconds=180),
        capture_limits=dict(available_before=11*1024**3, tmpfs_before=2*1024**3, rss=12*1024**3, seconds=900),
        minimum_free=1024**3, output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle / 'spec.json', spec)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    for path in TOOLS.glob('*.py'):
        ast.parse(path.read_text(encoding='utf8'), str(path))
    helpers = [COMMON, TOOLS.parent / 'managed-phase-amd/run.py', TOOLS.parent / 'ort-diagnosis-amd/run.py']
    write(BASE / 'prepared.json', dict(archive=pin(BASE / 'payload.tar.gz'), spec=pin(bundle / 'spec.json'),
        tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={p.relative_to(ROOT).as_posix(): pin(p) for p in helpers}))
    print(json.dumps(dict(prepared=True, archive=pin(BASE / 'payload.tar.gz'), consumer_rebuilt=False)))


def prepared():
    prerequisites()
    value = read(BASE / 'prepared.json')
    assert value['archive'] == pin(BASE / 'payload.tar.gz') and value['spec'] == pin(BASE / 'bundle/spec.json')
    for name, wanted in value['tools'].items():
        assert pin(TOOLS / name) == wanted, name
    for name, wanted in value['helpers'].items():
        assert pin(ROOT / name) == wanted, name
    for name, wanted in read(BASE / 'bundle/spec.json')['files'].items():
        assert pin(BASE / 'bundle' / name) == wanted, name
    for name, wanted in read(BASE / 'bundle/spec.json')['diagnostic_references'].items():
        assert pin(ROOT / name) == wanted, name
    for name, wanted in read(BASE / 'bundle/spec.json')['isolated_evidence'].items():
        assert pin(ROOT / name) == wanted, name


def observe(kind):
    result = ssh(PRELUDE + f'''
from remote import read,live
path=base/({kind!r}+'-state.json');state=read(path) if path.exists() else None
ids=[{read(BASE / (kind + '-deployment.json'))!r}]
if state:ids += [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
print(json.dumps(dict(live=[i for i in ids if live(i)],complete=state and state['complete'],code=state and state['code'],
 latest=None if not state or not state['runs'] else {{k:state['runs'][-1].get(k) for k in ['name','samples','complete','code']}},
 error=state and state.get('error'),stderr=(base/({kind!r}+'-supervisor.stderr')).read_text()[-4000:])))
''')
    with (BASE / (kind + '-observations.jsonl')).open('a') as stream:
        stream.write(json.dumps(result) + '\n')
    print(json.dumps(result))


def collect(kind):
    target = BASE / (kind + '-collected')
    assert not target.exists()
    script = PRELUDE + f'''
from remote import read,live,pin,verify
kind={kind!r};state=read(base/(kind+'-state.json'))
ids=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert state['complete'] and not any(live(i) for i in ids)
spec=verify();paths={{base/name for name in spec['files']}}
for folder in ['logs','inventory','runtime-original','runtime-control','runtime-observed']+(['clock','stages','markers'] if kind=='capture' else []):
 paths.update(p for p in (base/folder).rglob('*') if p.is_file())
paths.update(p for p in base.iterdir() if p.is_file() and p.name!='transfer.tar.gz')
files={{p.relative_to(base).as_posix():pin(p) for p in sorted(paths)}}
receipt=base/(kind+'-collection.json')
with receipt.open('x') as stream:json.dump(dict(files=files,state=pin(base/(kind+'-state.json')),terminal=True,code=state['code'],identities=ids),stream)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
 for name in [*files,receipt.name]:tar.add(base/name,arcname=name,recursive=False)
'''
    archive = BASE / (kind + '-results.tar.gz')
    with archive.open('xb') as out, (BASE / (kind + '-collection.stderr')).open('x') as err:
        result = subprocess.run(SSH + ['python3 -B -'], input=script.encode(), stdout=out, stderr=err,
            timeout=300, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, 'Preserve incomplete collection; do not repeat successful workload'
    target.mkdir()
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members}) == len(members)
        tar.extractall(target, filter='data')
    receipt = read(target / (kind + '-collection.json'))
    for name, wanted in receipt['files'].items():
        assert pin(target / name) == wanted, name
    write(BASE / (kind + '-transfer.json'), dict(passed=True, archive=pin(archive), collection=pin(target / (kind + '-collection.json'))))
    print(json.dumps(dict(code=receipt['code'], files=len(receipt['files']))))


if __name__ == '__main__':
    action = sys.argv[1]
    assert action in ['prepare', 'stage', 'launch', 'observe', 'collect']
    if action == 'prepare':
        prepare()
    else:
        prepared()
        if action == 'stage':
            transport.stage()
        else:
            kind = sys.argv[2]
            assert kind in ['build', 'capture']
            if action == 'observe':
                observe(kind)
            elif action == 'collect':
                collect(kind)
            else:
                if kind == 'capture':
                    assert read(BASE / 'build-review.json')['passed'] and read(BASE / 'build-review-transferred.json')['passed']
                transport.launch(kind)
