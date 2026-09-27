"""Reuse the bounded depthwise transport for current pointwise attribution."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from source import CONV, MATMUL, HELPER, changed

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-pointwise-cost-amd-20260927'
REMOTE = '/dev/shm/lokad-pointwise-cost-20260927'
PROFILE = ROOT/'artifacts/parakeet-decoder-lstm-layout-profile-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-lstm-layout-root-amd-20260927'
APP = ROOT/'artifacts/parakeet-decoder-lstm-layout-app-amd-20260927'


def load(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(value)
    return value


prior = load('pointwise_transport', TOOLS.parent/'weight-ownership-probe/run.py')
pin, read, write, ssh = prior.pin, prior.read, prior.write, prior.ssh
PRELUDE = prior.PRELUDE.replace(prior.REMOTE, REMOTE)
prior.BASE, prior.REMOTE, prior.PRELUDE = BASE, REMOTE, PRELUDE
transport = prior.transport
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE


def references():
    for folder, digest in [(QUALIFIED, 'efb99eea455647c64bbe0811c1b7d46387add59049ac81e48a926b250e7c42da'),
                           (PROFILE, '3f38a6a91df3e92e3d70dcf6eb79071fcedf37e1230c49133feb6099a75cc98a')]:
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json')
        assert proof['passed'] and proof['analysis'] == pin(folder/'analysis.json')
        names = ['bundle/stage.json'] if folder == QUALIFIED else ['bundle/spec.json','capture-collected/control/result.json']
        for name in names: assert pin(folder/name) == proof['files'][name], name
    spec = read(PROFILE/'bundle/spec.json')
    for name, wanted in spec['files'].items():
        assert pin(PROFILE/'bundle'/name) == wanted, name
    stage = read(QUALIFIED/'bundle/stage.json')
    source = {n.removeprefix('source/'):v for n,v in stage['files'].items() if n.startswith('source/')}
    assert len(source) == 443
    for name, wanted in source.items():
        assert pin(QUALIFIED/'bundle/source'/name) == wanted, name
        assert pin(ROOT/name) == wanted, name
    return source, spec


def prepare():
    assert not BASE.exists()
    source, old = references()
    values = {n:(QUALIFIED/'bundle/source'/n).read_bytes() for n in source}
    for name in [CONV, MATMUL]:
        values[name] = changed(name, values[name])
    values[HELPER] = (TOOLS/'PointwiseCostProbe.cs.txt').read_bytes()
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(encoding='utf8'))
    assert (TOOLS/'audit.py').is_file()
    BASE.mkdir()
    bundle = BASE/'bundle'
    bundle.mkdir()
    def put(name, data):
        path = bundle/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    for name, data in values.items():
        put('source/'+name, data)
    for name in ['Program.cs', 'Bridge.csproj', 'global.json']:
        put('bridge-source/'+name, (QUALIFIED/'bundle/bridge-source'/name).read_bytes())
    for target, origin in [('remote.py', TOOLS/'vm.py'),
        ('base_build.py', TOOLS.parent/'depthwise-route-amd/vm.py'),
        ('common.py', TOOLS.parent/'managed-phase-amd/remote.py'),
        ('campaign_processes.py', APP/'collected/runtime/campaign_processes.py'),
        ('protocol.py', APP/'collected/runtime/protocol.py'),
        ('reference-public.json', PROFILE/'capture-collected/control/result.json'),
        ('manifest.json', APP/'collected/manifests/current-parakeet.json'),
        ('protocol.md', TOOLS/'README.md'), ('prospective-plan.md', ROOT/'PLAN.md')]:
        put(target, origin.read_bytes())
    runtime = {n.removeprefix('runtime-control/'):v for n,v in old['files'].items() if n.startswith('runtime-control/')}
    previous = '/dev/shm/lokad-lstmlayout-profile-20260927/runtime-control'
    external = dict(old['external'])
    external.update({previous+'/'+n:v for n,v in runtime.items()})
    spec = dict(boot=old['boot'], external=external, runtime=runtime, prior=previous, app=old['app'],
        qualified_root=pin(QUALIFIED/'closed.json'), profile=pin(PROFILE/'closed.json'),
        source_before=source, changed_methods=['Conv2DFloatCore','MatMul2DCore','RunFloatMatMulKernel'],
        before_product={n:runtime[n] for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}, consumer=runtime['SampledAudio.dll'],
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3,tmpfs_before=1024**3,rss=3*1024**3,seconds=180),
        capture_limits=old['capture_limits'], minimum_free=old['minimum_free'], output_limit=old['output_limit'],
        observer_over_control_limit=1.05, diagnostic_only=True, release_admitted=False,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json', spec)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    helpers = ['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py',
        'ort-diagnosis-amd/run.py','feed-forward-cost-diagnostic/isolated_baseline.py','depthwise-route-amd/vm.py']
    write(BASE/'prepared.json', dict(archive=pin(BASE/'payload.tar.gz'),spec=pin(bundle/'spec.json'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={n:pin(TOOLS.parent/n) for n in helpers}))
    print(json.dumps(dict(prepared=True, archive=pin(BASE/'payload.tar.gz'), spec=pin(bundle/'spec.json'))))


def prepared():
    references()
    value = read(BASE/'prepared.json')
    assert value['archive'] == pin(BASE/'payload.tar.gz') and value['spec'] == pin(BASE/'bundle/spec.json')
    for name, wanted in value['tools'].items(): assert pin(TOOLS/name) == wanted, name
    for name, wanted in value['helpers'].items(): assert pin(TOOLS.parent/name) == wanted, name
    for name, wanted in read(BASE/'bundle/spec.json')['files'].items(): assert pin(BASE/'bundle'/name) == wanted, name


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare': prepare()
    else:
        prepared()
        if action == 'stage': transport.stage()
        else:
            kind = sys.argv[2]
            assert kind in ['build','capture']
            if action == 'launch':
                if kind == 'capture': assert read(BASE/'build-review-transferred.json')['passed']
                transport.launch(kind)
            else: {'observe':prior.observe, 'collect':prior.collect}[action](kind)
