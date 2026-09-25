"""Freeze one packed final-row proof against the corrected M76 binaries."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tarfile
from cases import MODES, cases

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
loader = importlib.util.spec_from_file_location('ownership_transport', TOOLS.parent/'weight-ownership-probe/run.py')
prior = importlib.util.module_from_spec(loader); loader.loader.exec_module(prior)
pin, read, write, ssh = prior.pin, prior.read, prior.write, prior.ssh
BASE = ROOT/'artifacts/parakeet-packed-final-row-proof-amd-20260925'
REMOTE = '/dev/shm/lokad-parakeet-packed-final-row-proof-20260925'
PRELUDE = prior.PRELUDE.replace(prior.REMOTE, REMOTE)
prior.BASE, prior.REMOTE, prior.PRELUDE = BASE, REMOTE, PRELUDE
transport = prior.transport
transport.BASE, transport.REMOTE, transport.PRELUDE = BASE, REMOTE, PRELUDE
MODELS = ROOT/'artifacts/parakeet-owned-packed-weight-models-amd-20260925'
DIAGNOSTIC = ROOT/'artifacts/parakeet-owned-packed-weight-reconstruction-cost-amd-20260925'
RUNTIME = '/dev/shm/lokad-parakeet-owned-packed-weight-models-20260925/runtimes/candidate'
SOURCES = ['Program', 'PackedFinalRowKernel', 'ExistingKernels']


def references():
    assert pin(MODELS/'closed.json')['sha256'] == 'bb1a758c6936fa251d921992c67dd7c65ff075fe20829af72f4455b529f1b2ca'
    model = read(MODELS/'closed.json'); assert model['passed'] and model['analysis'] == pin(MODELS/'analysis.json')
    for name, wanted in model['files'].items(): assert pin(MODELS/name) == wanted, name
    assert pin(DIAGNOSTIC/'closed.json')['sha256'] == '1750ff9e337a06840f1001b335ac211909bc721c09abc33f8aabc389e46c8f38'
    diagnostic = read(DIAGNOSTIC/'closed.json')
    assert diagnostic['passed'] and not diagnostic['usable_for_attribution']
    assert diagnostic['analysis'] == pin(DIAGNOSTIC/'analysis.json')
    result = read(DIAGNOSTIC/'analysis.json')
    assert sum(c['passed'] for c in result['summary']['controls']) == 89
    product = read(MODELS/'analysis.json')['identities']['candidate']
    assert product == result['products']['candidate']
    assert product['Lokad.Onnx.dll']['sha256'] == '82c02785506b540d3fe590d48b0fdb15fc4b744ffb376cd626d758d5755bb16f'
    original = (TOOLS.parent/'packed-logical-weight-probe-v2/PackedLogicalWeight.cs.txt').read_text()
    marker = 'internal static unsafe class ExistingKernels\n{'
    assert original.count(marker) == 1
    assert (TOOLS/'ExistingKernels.cs.txt').read_text() == 'using System.Reflection;\nusing Lokad.Onnx;\n\n' + marker + original.split(marker)[1]
    return product, result['failed_release_controls']


def prepare():
    assert not BASE.exists() and (TOOLS/'review.py').is_file()
    product, failed_controls = references()
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(), str(p))
    BASE.mkdir(); bundle = BASE/'bundle'; (bundle/'source').mkdir(parents=True)
    for name in SOURCES: (bundle/'source'/(name+'.cs')).write_bytes((TOOLS/(name+'.cs.txt')).read_bytes())
    (bundle/'source/global.json').write_bytes((ROOT/'global.json').read_bytes())
    names = ['Lokad.Onnx','Lokad.Onnx.Data','Google.Protobuf','FastBertTokenizer','Lokad.Tokenizers','SixLabors.ImageSharp']
    refs = ''.join(f'<Reference Include="{n}"><HintPath>$(FrozenProductDirectory)/{n}.dll</HintPath></Reference>' for n in names)
    project = '<Project Sdk="Microsoft.NET.Sdk"><PropertyGroup><OutputType>Exe</OutputType><TargetFramework>net10.0</TargetFramework><Nullable>enable</Nullable><ImplicitUsings>enable</ImplicitUsings><AllowUnsafeBlocks>true</AllowUnsafeBlocks><LangVersion>11.0</LangVersion><UseAppHost>false</UseAppHost></PropertyGroup><ItemGroup>' + refs + '</ItemGroup></Project>'
    (bundle/'source/PackedFinalRowProbe.csproj').write_text(project)
    for name, path in [('remote.py', TOOLS/'vm.py'), ('common.py', TOOLS.parent/'managed-phase-amd/remote.py'),
                       ('plan.md', ROOT/'.agent/m78-parakeet-packed-final-row-20260925.md')]:
        (bundle/name).write_bytes(path.read_bytes())
    runtime = {n+'.dll': pin(MODELS/'collected/runtimes/candidate'/(n+'.dll')) for n in names}
    assert all(runtime[name] == wanted for name, wanted in product.items())
    spec = dict(boot=1789634288.0, original_runtime=RUNTIME, runtime_files=runtime, product=product,
                model_closure=pin(MODELS/'closed.json'), diagnostic_closure=pin(DIAGNOSTIC/'closed.json'),
                prior_quantitative_attribution=False, failed_release_controls=failed_controls,
                release_admitted=False, diagnostic_only=True, no_model_execution=True, cases={mode: cases(mode) for mode in MODES},
                external={RUNTIME+'/'+name: value for name, value in runtime.items()},
                feed='/dev/shm/lokad-pyannote-blocked-spatial-app-20260922/nuget-feed',
                build_limits=dict(available_before=2*1024**3, tmpfs_before=1024**3, rss=2*1024**3, seconds=180),
                capture_limits=dict(available_before=2*1024**3, tmpfs_before=1024**3, rss=2*1024**3, seconds=300),
                minimum_free=1024**3, output_limit=64*1024**2,
                files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json', spec)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for p in bundle.rglob('*'):
            if p.is_file(): archive.add(p, arcname=p.relative_to(bundle).as_posix(), recursive=False)
    helpers = ['weight-ownership-probe/run.py','managed-phase-amd/run.py','managed-phase-amd/remote.py',
               'ort-diagnosis-amd/run.py','packed-logical-weight-probe-v2/PackedLogicalWeight.cs.txt']
    write(BASE/'prepared.json', dict(archive=pin(BASE/'payload.tar.gz'), spec=pin(bundle/'spec.json'),
        tools={p.name: pin(p) for p in TOOLS.iterdir() if p.is_file()}, helpers={name: pin(TOOLS.parent/name) for name in helpers}))
    print(json.dumps(dict(prepared=True, archive=pin(BASE/'payload.tar.gz'), spec=pin(bundle/'spec.json'), cases={mode: len(cases(mode)) for mode in MODES})))


def prepared():
    product, failed_controls = references(); value = read(BASE/'prepared.json'); spec = read(BASE/'bundle/spec.json')
    assert value['archive'] == pin(BASE/'payload.tar.gz') and value['spec'] == pin(BASE/'bundle/spec.json')
    for name, wanted in value['tools'].items(): assert pin(TOOLS/name) == wanted, name
    for name, wanted in value['helpers'].items(): assert pin(TOOLS.parent/name) == wanted, name
    for name, wanted in spec['files'].items(): assert pin(BASE/'bundle'/name) == wanted, name
    assert spec['product'] == product and spec['failed_release_controls'] == failed_controls


if __name__ == '__main__':
    action = sys.argv[1]
    if action == 'prepare': prepare()
    else:
        prepared()
        if action == 'stage': transport.stage()
        else:
            kind = sys.argv[2]; assert kind in ['build','capture']
            if action == 'launch':
                if kind == 'capture': assert read(BASE/'build-review-transferred.json')['passed']
                transport.launch(kind)
            else: {'observe': prior.observe, 'collect': prior.collect}[action](kind)
