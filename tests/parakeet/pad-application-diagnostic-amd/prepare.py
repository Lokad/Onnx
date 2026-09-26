"""Freeze one diagnostic using exact rejected binaries and retained observers."""
import ast
import json
import shutil
import tarfile
from pathlib import Path
from compatibility import ROOT, SELECTED, CANDIDATE, OBSERVER, CONSUMER, review
from protocol import pin, read, save
from scope import consumer, supervisor, attribute_module, replace_once

TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-pad-application-diagnostic-amd-20260926'
APP = ROOT/'artifacts/parakeet-owned-batch-isolation-release-app-amd-20260925'
REMOTE_APP = '/dev/shm/lokad-parakeet-owned-batch-isolation-release-app-20260925'
EVENTS = ROOT/'artifacts/e5-direct-tier-diagnostic-v2-amd-20260925'
REMOTE_EVENTS = '/dev/shm/lokad-e5-direct-tier-diagnostic-v2-20260925'


def previous_closed():
    result = review()
    assert read(APP/'closed.json')['passed'] and read(APP/'closed.json')['admitted']
    assert pin(APP/'closed.json')['sha256'] == 'e2ddd739372df25f93323a717f97b8cc08999faa143f67ede16130e32b572f03'
    assert pin(EVENTS/'closed.json')['sha256'] == 'e61941d9308e33b688f222665a0f79dd985f87b85d56123554a47202d8c531bc'
    assert read(EVENTS/'closed.json')['passed']
    assert read(EVENTS/'payload.json')['reused_exporter']['roundtrip']['passed']
    return result


def prepare():
    assert not BASE.exists()
    compatible = previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    inputs = dict(compatible['inputs'])
    def copy(source, name):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target); inputs[source.relative_to(ROOT).as_posix()] = pin(source)
    def derive(source, name, transform):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(transform(source.read_text(encoding='utf8')), encoding='utf8')
        inputs[source.relative_to(ROOT).as_posix()] = pin(source)
    for path in (CONSUMER/'bundle/consumer-source').iterdir():
        if path.name == 'Program.cs': derive(path, 'source/consumer/'+path.name, consumer)
        elif path.name == 'SampledAudio.csproj':
            derive(path, 'source/consumer/'+path.name,
                   lambda s: replace_once(s, '</ItemGroup>', '<Compile Include="TraceStartup.cs"/></ItemGroup>'))
        else: copy(path, 'source/consumer/'+path.name)
    copy(TOOLS/'TraceStartup.cs', 'source/consumer/TraceStartup.cs')
    copy(ROOT/'global.json', 'source/global.json')
    derive(ROOT/'tests/parakeet/selected-profile-build-amd/Bridge.cs.txt', 'source/bridge/Program.cs',
           lambda s: replace_once(s, 'new[] { "SampledAudio.dll" }',
                                  'new[] { "Lokad.Onnx.dll", "Lokad.Onnx.Data.dll", "SampledAudio.dll" }'))
    copy(ROOT/'tests/parakeet/selected-profile-build-amd/Bridge.csproj', 'source/bridge/Bridge.csproj')
    for name in ['protocol.py', 'remote.py', 'remote_prepare.py', 'scope.py']:
        copy(TOOLS/name, 'tools/'+name)
    derive(ROOT/'tests/parakeet/dispatch-events-amd/remote.py', 'tools/remote_base.py', supervisor)
    copy(APP/'collected/runtime/protocol.py', 'tools/application_protocol.py')
    derive(ROOT/'tests/parakeet/packed-final-row-profile-amd/audit.py', 'tools/phase_audit.py', attribute_module)
    copy(TOOLS/'README.md', 'prospective-plan.md')
    for name in ['closed.json', 'collected/collection.json']:
        copy(EVENTS/name, 'evidence/events-'+Path(name).name)
    copy(OBSERVER/'build-collected/inventory/instructions.json', 'evidence/observer-instructions.json')
    copy(APP/'collected/manifests/current-parakeet.json', 'manifest.json')
    save(bundle/'compatibility.json', compatible)
    current = SELECTED/'collected/runtimes/current'
    dependencies = ['Google.Protobuf.dll', 'FastBertTokenizer.dll', 'Lokad.Tokenizers.dll', 'SixLabors.ImageSharp.dll']
    for role in ['reference', 'current', 'candidate']:
        for name in dependencies: copy(current/name, 'runtimes/'+role+'/'+name)
        copy((CANDIDATE/'collected/runtime' if role == 'candidate' else current)/'Lokad.Onnx.dll',
             'runtimes/'+role+'/Lokad.Onnx.dll')
        copy((current if role == 'reference' else OBSERVER/'build-collected/runtime-observed')/'Lokad.Onnx.Data.dll',
             'runtimes/'+role+'/Lokad.Onnx.Data.dll')
    for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
        copy(CONSUMER/'build-collected/runtime-control'/('SampledAudio.'+suffix),
             'runtimes/reference/SampledAudio.'+suffix)
    links = {}
    manifest = read(bundle/'manifest.json')
    assert manifest['warmup_passes'] == 1 and manifest['measured_passes'] == 3 and len(manifest['cases']) == 20
    for value in [manifest['reference'], *[c['pcm'] for c in manifest['cases']]]:
        links['assets/'+value['path']] = dict(source=REMOTE_APP+'/assets/'+value['path'],
                                             identity={k:value[k] for k in ['bytes','sha256']})
    external = {v['path']:{k:v[k] for k in ['bytes','sha256']} for v in manifest['models'].values()}
    events_payload = read(EVENTS/'payload.json'); events_built = read(EVENTS/'collected/built.json')
    event_files = read(EVENTS/'collected/collection.json')['files']
    assert events_built['exporter'] == events_payload['reused_exporter']['binary'] == event_files['export-runtime/DispatchEventsExport.dll']
    for mapping, prefix in [(events_payload['files'], 'tracer/'), (event_files, 'export-runtime/')]:
        for name, wanted in mapping.items():
            if name.startswith(prefix): links[name] = dict(source=REMOTE_EVENTS+'/'+name, identity=wanted)
    candidate = read(CANDIDATE/'collected/inventory/instructions.json')['observations'][0]
    save(bundle/'stage.json', dict(passed=True, diagnostic_only=True, links=links,
        products=compatible['products'], observer=compatible['observer'], original_consumer=compatible['consumer'],
        candidate_methods=candidate['candidate_methods'], external=external,
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        interpreter=events_payload['interpreter'], exporter=events_built['exporter'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for path in TOOLS.iterdir():
        if path.is_file():
            if path.suffix == '.py': ast.parse(path.read_text(encoding='utf8'), str(path))
            inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    inputs['tests/parakeet/pad-runtime-diagnostic-amd/run.py'] = pin(ROOT/'tests/parakeet/pad-runtime-diagnostic-amd/run.py')
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    save(BASE/'prepared.json', dict(passed=True, files=inputs, archive=pin(BASE/'payload.tar.gz'), stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(passed=True, archive=pin(BASE/'payload.tar.gz'), links=len(links))))


if __name__ == '__main__': prepare()
