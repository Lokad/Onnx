"""One fixed public-call comparison, reusing the qualified screen supervisor."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location('decoder_screen_transport', TOOLS.parent/'weight-ownership-probe/run.py')
prior = importlib.util.module_from_spec(loader); loader.loader.exec_module(prior)
pin, read, write, ssh = prior.pin, prior.read, prior.write, prior.ssh
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-screen-v2-amd-20260927'
REMOTE = '/dev/shm/lokad-decrow-screen2-20260927'
PRELUDE = prior.PRELUDE.replace(prior.REMOTE, REMOTE)
prior.BASE, prior.REMOTE, prior.PRELUDE = BASE, REMOTE, PRELUDE
transport = prior.transport
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE
FIRST = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-amd-20260927'
CONTRACTS = ROOT/'artifacts/parakeet-decoder-packed-row-contracts-v3-amd-20260927'
RUNTIMES = {role: FIRST/'collected/runtimes'/role for role in ['current', 'candidate']}
REMOTE_RUNTIMES = {role: '/dev/shm/lokad-decrow-20260927/runtimes/'+role for role in RUNTIMES}
HELPERS = ['weight-ownership-probe/run.py', 'managed-phase-amd/run.py', 'managed-phase-amd/remote.py',
    'ort-diagnosis-amd/run.py', 'feed-forward-cost-diagnostic/isolated_baseline.py',
    'prepared-recurrence-timing-amd/campaign_processes.py', 'vector-sigmoid-screen/vm.py', 'vector-sigmoid-screen/audit.py']


def census():
    cases = []
    for name, kind, n, k, weight, a_shape in [
        ('captured-projection', 'target', 640, 8198, 'onnx::MatMul_230', [1, 1, 1, 640]),
        ('captured-unmapped', 'unmapped', 640, 8198, 'onnx::MatMul_230', [1, 1, 1, 640]),
        ('captured-scalar', 'scalar', 640, 8198, 'onnx::MatMul_230', [1, 1, 1, 640]),
        ('captured-simd', 'simd', 640, 8198, 'onnx::MatMul_230', [1, 1, 1, 640]),
        ('prediction-width-control', 'narrow', 640, 640, 'onnx::MatMul_229', [1, 1, 640]),
        ('encoder-width-control', 'narrow', 1024, 640, 'onnx::MatMul_228', [1, 1, 1024])]:
        cases.append(dict(index=len(cases), name=name, kind=kind, reduction=n, weight=weight,
            a_shape=a_shape, shape=[*a_shape[:-1], k], batch=max(1, min(256, 65536//k))))
    return dict(protocol='decoder-packed-row-public-600-180-v1', cases=cases, warmup_rounds=600, measured_rounds=180,
        order=['current', 'candidate', 'candidate', 'current'], target_max_ratio=.75, repeat_max_ratio=1.10,
        fallback_max_ratio=1.05, narrow_activations='synthetic: ((i*37+17)%101-50)*0.03125')


def references():
    failed = ROOT/'artifacts/parakeet-decoder-packed-row-screen-amd-20260927'
    assert pin(failed/'failed.json')['sha256'] == 'a5d1059582dedff385b9b1e814b5be365c5a9541178f1033920ff87aa398cef0'
    failure = read(failed/'failed.json'); assert failure['terminal'] and not failure['capture_executed']
    for name, wanted in failure['files'].items(): assert pin(failed/name) == wanted, name
    old = (failed/'frozen-tools/Screen.cs').read_text()
    before = '        Require(OperatingSystem.IsLinux() && args.Length == 4, "Linux: base role sequence output");'
    after = '        if (!OperatingSystem.IsLinux()) throw new PlatformNotSupportedException("Linux screen");\n        Require(args.Length == 4, "base role sequence output");'
    assert old.count(before) == 1 and (TOOLS/'Screen.cs').read_text() == old.replace(before, after)
    for name in ['audit.py', 'score.py', 'test_score.py']: assert pin(TOOLS/name) == pin(failed/'frozen-tools'/name)
    assert census() == read(failed/'bundle/census.json')
    path = CONTRACTS/'closed.json'
    assert pin(path)['sha256'] == 'fc00688c8ef0e65e5d5109f1808209add023e740aabc5b4312647e547df7d61d'
    closure = read(path); assert closure['passed']
    for name, wanted in closure['files'].items(): assert pin(CONTRACTS/name) == wanted, name
    report = read(TOOLS.parent/'decoder-packed-row-results/contracts-20260927.json')
    products = report['products']
    assert report['passed'] and report['public_cases'] == 270 and report['raw_cases'] == 162
    for role in products: assert products[role] == pin(RUNTIMES[role]/'Lokad.Onnx.dll')
    assert products['current']['sha256'] == '65f15a41764660af6166c9a965943b6f28cac0c9505117d0fd4beaa2687f9d03'
    assert products['candidate']['sha256'] == 'af19b3b4429a07f7966b5e35ee04e8a31316f45991c300f3683a591caf5e9374'
    return products, {'contracts': pin(path), 'contracts_report': pin(TOOLS.parent/'decoder-packed-row-results/contracts-20260927.json'),
        'prior_build_warning_failure': pin(failed/'failed.json')}


def prepare():
    assert not BASE.exists()
    products, evidence = references()
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(encoding='utf8'))
    for name in ['Screen.cs', 'README.md', 'score.py', 'audit.py', 'test_score.py']: assert (TOOLS/name).is_file()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    def put(name, content):
        p = bundle/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(content)
    put('source/Screen.cs', (TOOLS/'Screen.cs').read_bytes())
    put('source/global.json', (ROOT/'global.json').read_bytes())
    refs = ''.join(f'<Reference Include="{n}"><HintPath>$(FrozenProductDirectory)/{n}.dll</HintPath></Reference>' for n in ['Lokad.Onnx', 'Google.Protobuf'])
    project = '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><ImplicitUsings>enable</ImplicitUsings><Nullable>enable</Nullable><UseAppHost>false</UseAppHost></PropertyGroup><ItemGroup>'+refs+'</ItemGroup></Project>'
    put('source/Screen.csproj', project.encode())
    put('common.py', (TOOLS.parent/'managed-phase-amd/remote.py').read_bytes())
    put('remote.py', (TOOLS.parent/'vector-sigmoid-screen/vm.py').read_bytes())
    put('campaign_processes.py', (TOOLS.parent/'prepared-recurrence-timing-amd/campaign_processes.py').read_bytes())
    put('protocol.md', (TOOLS/'README.md').read_bytes())
    put('fixture.json', (CONTRACTS/'bundle/fixture.json').read_bytes())
    put('projection-a.f32', (CONTRACTS/'bundle/projection-a.f32').read_bytes())
    write(bundle/'census.json', census())
    fixture = read(bundle/'fixture.json'); stage = read(CONTRACTS/'bundle/stage.json')
    assert stage['model']['sha256'] == fixture['model_sha256']
    assert pin(bundle/'projection-a.f32')['sha256'] == fixture['a_sha256']
    external = {REMOTE_RUNTIMES[role]+'/Lokad.Onnx.dll': products[role] for role in products}
    protobuf = pin(RUNTIMES['current']/'Google.Protobuf.dll')
    assert pin(RUNTIMES['candidate']/'Google.Protobuf.dll') == protobuf
    external[REMOTE_RUNTIMES['current']+'/Google.Protobuf.dll'] = protobuf
    external[fixture['model']] = stage['model']
    spec = dict(boot=1789634288.0, products=products, evidence=evidence, external=external, runtimes=REMOTE_RUNTIMES,
        diagnostic_only=True, release_admitted=False, census=pin(bundle/'census.json'),
        feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
        build_limits=dict(available_before=2*1024**3, tmpfs_before=1024**3, rss=3*1024**3, seconds=180),
        capture_limits=dict(available_before=2*1024**3, tmpfs_before=1024**3, rss=3*1024**3, seconds=900),
        minimum_free=1024**3, output_limit=32*1024**2,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json', spec)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    write(BASE/'prepared.json', dict(archive=pin(BASE/'payload.tar.gz'), spec=pin(bundle/'spec.json'),
        tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()},
        helpers={(TOOLS.parent/n).relative_to(ROOT).as_posix(): pin(TOOLS.parent/n) for n in HELPERS}))
    print(json.dumps(dict(prepared=True, archive=pin(BASE/'payload.tar.gz'), cases=len(census()['cases']))))


def prepared():
    products, evidence = references(); value = read(BASE/'prepared.json'); spec = read(BASE/'bundle/spec.json')
    assert spec['products'] == products and spec['evidence'] == evidence
    assert value['archive'] == pin(BASE/'payload.tar.gz') and value['spec'] == pin(BASE/'bundle/spec.json')
    for name, wanted in value['tools'].items(): assert pin(TOOLS/name) == wanted, name
    for name, wanted in value['helpers'].items(): assert pin(ROOT/name) == wanted, name
    for name, wanted in spec['files'].items(): assert pin(BASE/'bundle'/name) == wanted, name
    assert read(BASE/'bundle/census.json') == census()


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare': prepare()
    else:
        prepared()
        if action == 'stage': transport.stage()
        else:
            kind = sys.argv[2]; assert kind in ['build', 'capture']
            if action == 'launch':
                if kind == 'capture': assert read(BASE/'build-review-transferred.json')['passed']
                transport.launch(kind)
            else: {'observe': prior.observe, 'collect': prior.collect}[action](kind)
