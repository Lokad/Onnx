"""Bind the unchanged observer, capture and audit to the qualified padding root."""
import ast
import json
import shutil
import tarfile
from run import ROOT, TOOLS, PARENT, ORIGINAL, BASE, REMOTE_APP, APP, pin, read, write
from reuse import OLD, QUALIFIED_ROOT, review_observer

SOURCE = ROOT / 'artifacts/parakeet-pad-current-source-20260926'
QUALIFICATION = dict(app=APP, root=QUALIFIED_ROOT,
    pyannote=ROOT / 'artifacts/parakeet-pad-current-pyannote-app-amd-20260926',
    graphs=ROOT / 'artifacts/parakeet-pad-current-graphs-v2-amd-20260926')


def diagnostic_gates():
    observer = review_observer()
    for label, folder in QUALIFICATION.items():
        proof = read(folder / 'closed.json')
        assert proof['passed']
        if label != 'root':
            assert proof['admitted']
        for name, wanted in proof['files'].items():
            assert pin(folder / name) == wanted, name
    assert pin(APP / 'closed.json')['sha256'] == '2b40b54d8e94de7326ceec5228ad1529c7513964211e8b5dfbb2c921cd798151'
    assert pin(QUALIFICATION['graphs'] / 'closed.json')['sha256'] == 'bea082c63f1e4497f2eacfc3e6bd77325e63b4367c6ce40e3fb3fe69964cb96f'
    app = read(APP / 'analysis.json')
    pyannote = read(QUALIFICATION['pyannote'] / 'analysis.json')
    graphs = read(QUALIFICATION['graphs'] / 'analysis.json')
    for value, controls, gates in [(app, 63, 21), (pyannote, 12, 4)]:
        assert value['performance']['admitted']
        assert len(value['performance']['controls']) == controls
        assert len(value['performance']['gates']) == gates
        assert all(r['passed'] for kind in ['controls', 'gates'] for r in value['performance'][kind])
        assert value['identities']['candidate'] == observer['reference_product']
    assert app['identities']['current'] == pyannote['identities']['selected'] == observer['parent_product']
    assert len(graphs['performance']) == 8 and all(r['qualified'] for r in graphs['performance'])
    assert graphs['products']['candidate']['Lokad.Onnx.dll'] == observer['reference_product']['Lokad.Onnx.dll']
    assert graphs['products']['current']['Lokad.Onnx.dll'] == observer['parent_product']['Lokad.Onnx.dll']
    root = read(QUALIFIED_ROOT / 'analysis.json')
    assert root['measured'] == observer['reference_product'] and root['built'] == observer['product']
    assert root['root_source_verified']
    applied = read(QUALIFIED_ROOT / 'bundle/evidence/root-applied.json')
    assert applied['prerequisites']['parakeet-app'] == pin(APP / 'closed.json')
    assert applied['prerequisites']['pyannote-app'] == pin(QUALIFICATION['pyannote'] / 'closed.json')
    assert applied['graph_qualification'] == pin(QUALIFICATION['graphs'] / 'closed.json')
    assert len(applied['source_files']) == 437
    for name, wanted in applied['source_files'].items():
        assert pin(ROOT / name) == wanted, name
    prefixes = ['src/', 'tests/Lokad.Onnx.Backend.Tests/', 'tests/Lokad.Onnx.Tensors.Tests/']
    actual = {p.relative_to(ROOT).as_posix() for prefix in prefixes for p in (ROOT / prefix).rglob('*')
              if p.is_file() and not {'bin', 'obj'} & set(p.relative_to(ROOT).parts)}
    assert actual == {n for n in applied['source_files'] if any(n.startswith(prefix) for prefix in prefixes)}
    assert pin(SOURCE / 'prepared.json')['sha256'] == 'f9eb6c4a26362529d4fe19089e35201d5e26f318561445b58078d05c1df5dbea'
    for name, wanted in read(SOURCE / 'prepared.json')['source'].items():
        assert pin(SOURCE / 'source' / name) == wanted, name
    reference = read(APP / 'collected/evidence/candidate-public.json')
    assert reference['core_sha256'] == observer['reference_product']['Lokad.Onnx.dll']['sha256']
    assert reference['data_sha256'] == observer['reference_product']['Lokad.Onnx.Data.dll']['sha256']
    assert len({r['name'] for r in reference['records']}) == 20
    return observer


def prepare():
    assert not BASE.exists()
    observer = diagnostic_gates()
    BASE.mkdir()
    bundle = BASE / 'bundle'
    bundle.mkdir()
    inputs = dict(observer['inputs'])

    def copy(path, name):
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)

    copy(ORIGINAL / 'remote.py', 'remote.py')
    copy(TOOLS / 'README.md', 'prospective-plan.md')
    copy(APP / 'collected/evidence/candidate-public.json', 'evidence/candidate-public.json')
    for label, folder in QUALIFICATION.items():
        for name in ['closed.json', 'analysis.json']:
            copy(folder / name, f'evidence/qualification/{label}/{name}')
    for name in ['build-review.json', 'build-collected/built.json', 'build-collected/inventory/instructions.json']:
        copy(OLD / name, 'evidence/retained-observer/' + name)
    copy(SOURCE / 'prepared.json', 'evidence/product-source.json')
    copy(QUALIFIED_ROOT / 'bundle/evidence/root-applied.json', 'evidence/root-applied.json')
    copy(QUALIFIED_ROOT / 'collected/inventory/instructions.json', 'evidence/root-instructions.json')
    for mode in ['runtime-control', 'runtime-observed']:
        for path in (OLD / 'build-collected' / mode).iterdir():
            source = path
            if path.name == 'Lokad.Onnx.dll' or (mode == 'runtime-control' and path.name == 'Lokad.Onnx.Data.dll'):
                source = QUALIFIED_ROOT / 'collected/runtime' / path.name
            copy(source, mode + '/' + path.name)
    built = dict(core=observer['product']['Lokad.Onnx.dll'], data=observer['observed_data'],
        consumer=observer['consumer'], observer_rebuilt=False,
        runtime_files={p.relative_to(bundle).as_posix(): pin(p)
                       for mode in ['runtime-control', 'runtime-observed'] for p in (bundle / mode).iterdir()})
    write(bundle / 'built.json', built)
    review = dict(passed=True, built=pin(bundle / 'built.json'), core_unchanged=True,
        consumer_unchanged=True, constructor_unchanged=True,
        **{k: v for k, v in observer.items() if k not in ['passed', 'inputs']})
    write(bundle / 'build-review.json', review)
    shutil.copy2(bundle / 'build-review.json', BASE / 'build-review.json')
    manifest = read(APP / 'collected/manifests/current-parakeet.json')
    external = {}
    for value in manifest['models'].values():
        external[value['path']] = {k: value[k] for k in ['bytes', 'sha256']}
    for value in [manifest['reference'], *[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP + '/assets/' + value['path']] = {k: value[k] for k in ['bytes', 'sha256']}
    for name in ['manifests/current-parakeet.json', 'runtime/protocol.py', 'runtime/campaign_processes.py']:
        path = APP / 'collected' / name
        external[REMOTE_APP + '/' + name] = pin(path)
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    spec = dict(boot=1789634288.0, app=REMOTE_APP, external=external,
        source_receipt=pin(QUALIFIED_ROOT / 'bundle/evidence/root-applied.json'),
        measured_source_receipt=pin(SOURCE / 'prepared.json'), reference_product=observer['reference_product'],
        qualification_closures={k: pin(v / 'closed.json') for k, v in QUALIFICATION.items()},
        diagnostic_context=dict(isolated_candidate=False, release_admitted=True, root_product_changed=True,
            application_admitted=True, graph_admitted=True, graph_controls_passed=True,
            failed_graph_cases=[], observer_rebuilt=False, retained_observer_review=observer['original_review'],
            actual_root_binaries=True, root_metadata_equal=True, data_compiled_scope_equal=True),
        original_consumer=observer['consumer'], core=observer['product']['Lokad.Onnx.dll'],
        data=observer['product']['Lokad.Onnx.Data.dll'],
        capture_limits=dict(available_before=11 * 1024**3, tmpfs_before=2 * 1024**3,
                            rss=12 * 1024**3, seconds=900),
        minimum_free=1024**3, output_limit=512 * 1024**2,
        files={p.relative_to(bundle).as_posix(): pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle / 'spec.json', spec)
    for folder in [TOOLS, PARENT, ORIGINAL]:
        for path in folder.iterdir():
            if path.is_file():
                if path.suffix == '.py':
                    ast.parse(path.read_text(encoding='utf8'), str(path))
                inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    path = TOOLS.parent / 'ort-diagnosis-amd/run.py'
    inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE / 'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    value = dict(passed=True, archive=pin(BASE / 'payload.tar.gz'), spec=pin(bundle / 'spec.json'), inputs=inputs)
    write(BASE / 'prepared.json', value)
    print(json.dumps(dict(passed=True, archive=value['archive'], spec=value['spec'], observer_rebuilt=False)))
