"""Reuse the bounded depthwise transport for current attention attribution."""
import ast
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from source import PROVIDER, MATMUL, HELPER, changed

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-attention-cost-amd-20260927'
REMOTE = '/dev/shm/lokad-attention-cost-20260927'
PROFILE = ROOT/'artifacts/parakeet-pointwise-tail-profile-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-pointwise-tail-root-amd-20260927'
APP = ROOT/'artifacts/parakeet-pointwise-tail-app-amd-20260927'


def load(name, path):
    loader = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(value)
    return value


prior = load('attention_transport', TOOLS.parent/'weight-ownership-probe/run.py')
pin, read, write, ssh = prior.pin, prior.read, prior.write, prior.ssh
PRELUDE = prior.PRELUDE.replace(prior.REMOTE, REMOTE)
prior.BASE, prior.REMOTE, prior.PRELUDE = BASE, REMOTE, PRELUDE
transport = prior.transport
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE


def references():
    for folder, digest in [(QUALIFIED, 'fc11676361a50ef613f783fcb51e4ead488c9c2ee34a3edddcbb561f67037e47'),
                           (PROFILE, '378f7c1979052e4dcf06af4d2c9edad04cf899cc3d40de5f7a491f2c5f0c13b2')]:
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
    assert len(source) == 445
    for name, wanted in source.items():
        assert pin(QUALIFIED/'bundle/source'/name) == wanted, name
        assert pin(ROOT/name) == wanted, name
    return source, spec


def attention_metadata():
    results = TOOLS.parent/'pointwise-tail-profile-results'
    closed = ROOT/'artifacts/parakeet-attention-route-review-20260927/closed.json'
    assert pin(closed)['sha256'] == 'e83b34fb3aef0dacfc387c919638fd710fa266cd11a315fb6494195cd29ebc72'
    value = read(results/'attention-routes-20260927.json')
    assert value.pop('closure') == pin(closed)
    assert value == read(closed.parent/'analysis.json')
    assert pin(closed.parent/'analysis.json') == read(closed)['analysis']
    assert pin(results/'attention-clocks-20260927.csv') == read(closed)['clocks']
    projection = results/'projection-breakdown-20260927.json'
    assert pin(projection) == value['inputs'][projection.relative_to(ROOT).as_posix()]
    weights = {r['native']:r['weight']['name'] for r in read(projection)['records'] if r['group'].startswith('linear_')}
    with (results/'attention-clocks-20260927.csv').open(newline='') as stream:
        rows = [dict(request=int(r['request']),clip=r['clip'],pass_index=int(r['pass_index']),
            phase=r['phase'],node=r['node'],weight=weights[r['node']],m=int(r['rows']),route=r['route'])
            for r in csv.DictReader(stream)]
    assert len(rows) == 9600 and len(weights) == len(set(weights.values())) == 120
    return dict(closure=pin(closed),rows=rows)


def prepare():
    assert not BASE.exists()
    source, old = references()
    expected = attention_metadata()
    values = {n:(QUALIFIED/'bundle/source'/n).read_bytes() for n in source}
    for name in [PROVIDER, MATMUL]:
        values[name] = changed(name, values[name])
    values[HELPER] = (TOOLS/'AttentionCostProbe.cs.txt').read_bytes()
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
        data = origin.read_bytes()
        if target == 'base_build.py':
            # The current inspector requires its unchanged dependency directory.
            # Earlier pointwise work preserved/recovered this exact missing argument.
            before = "BASE/'logs/instructions.json'],env,BASE,limits,spec)"
            after = "BASE/'logs/instructions.json',spec['prior']],env,BASE,limits,spec)"
            text = data.decode()
            assert text.count(before) == 1
            adapted = text.replace(before, after)
            assert adapted.replace(after, before) == text
            data = adapted.encode()
        put(target, data)
    put('expected-attention.json', json.dumps(expected, indent=2).encode())
    runtime = {n.removeprefix('runtime-control/'):v for n,v in old['files'].items() if n.startswith('runtime-control/')}
    previous = '/dev/shm/lokad-pwt-profile-20260927/runtime-control'
    external = dict(old['external'])
    external.update({previous+'/'+n:v for n,v in runtime.items()})
    spec = dict(boot=old['boot'], external=external, runtime=runtime, prior=previous, app=old['app'],
        qualified_root=pin(QUALIFIED/'closed.json'), profile=pin(PROFILE/'closed.json'),
        source_before=source, changed_methods=['MatMul','RunIsolatedShortWidePackedRows'],
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
    assert read(BASE/'bundle/expected-attention.json') == attention_metadata()
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
