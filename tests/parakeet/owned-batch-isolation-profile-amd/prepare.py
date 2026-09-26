"""Reuse the closed observer and original capture after actual release admission."""
import ast
import json
import shutil
import tarfile
from run import ROOT, TOOLS, ORIGINAL, BASE, REMOTE_APP, APP, pin, read, write
from reuse import OLD, QUALIFIED_ROOT, review_observer

SOURCE = ROOT/'artifacts/parakeet-owned-batch-isolation-source-20260925'
QUALIFICATION = dict(app=APP,
    root=QUALIFIED_ROOT,
    pyannote=ROOT/'artifacts/parakeet-owned-batch-isolation-pyannote-app-amd-20260925',
    graphs=ROOT/'artifacts/parakeet-owned-batch-graph-qualification-20260925')


def diagnostic_gates():
    observer = review_observer()
    for label, folder in QUALIFICATION.items():
        proof = read(folder/'closed.json')
        assert proof['passed']
        if label != 'root': assert proof['admitted']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    root = read(QUALIFICATION['root']/'analysis.json')
    assert root['measured'] == observer['reference_product'] and root['root_source_verified']
    assert root['built'] == observer['product']
    applied = read(QUALIFICATION['root']/'bundle/evidence/root-applied.json')
    for name, wanted in applied['source_files'].items(): assert pin(ROOT/name) == wanted, name
    assert read(APP/'analysis.json')['identities']['candidate'] == observer['reference_product']
    assert read(QUALIFICATION['pyannote']/'analysis.json')['identities']['candidate'] == observer['reference_product']
    assert read(QUALIFICATION['graphs']/'analysis.json')['products']['candidate']['Lokad.Onnx.dll'] == observer['reference_product']['Lokad.Onnx.dll']
    source = read(SOURCE/'prepared.json')
    for name, wanted in source['source'].items(): assert pin(SOURCE/'source'/name) == wanted, name
    return observer


def prepare():
    assert not BASE.exists()
    observer = diagnostic_gates()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    inputs = dict(observer['inputs'])
    def copy(path, name):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target); inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    for name in ['remote.py']: copy(ORIGINAL/name, name)
    copy(TOOLS/'README.md', 'prospective-plan.md')
    copy(APP/'collected/evidence/candidate-public.json', 'evidence/candidate-public.json')
    for label, folder in QUALIFICATION.items():
        for name in ['closed.json','analysis.json']: copy(folder/name, f'evidence/qualification/{label}/{name}')
    for name in ['build-review.json','build-collected/built.json','build-collected/inventory/instructions.json']:
        copy(OLD/name, 'evidence/retained-observer/'+name)
    copy(SOURCE/'prepared.json', 'evidence/product-source.json')
    copy(QUALIFIED_ROOT/'bundle/evidence/root-applied.json', 'evidence/root-applied.json')
    copy(QUALIFIED_ROOT/'collected/inventory/instructions.json', 'evidence/root-instructions.json')
    for mode in ['runtime-control','runtime-observed']:
        for path in (OLD/'build-collected'/mode).iterdir():
            source = path
            if path.name == 'Lokad.Onnx.dll' or (mode == 'runtime-control' and path.name == 'Lokad.Onnx.Data.dll'):
                source = QUALIFIED_ROOT/'collected/runtime'/path.name
            copy(source, mode+'/'+path.name)
    built = dict(core=observer['product']['Lokad.Onnx.dll'], data=observer['observed_data'],
        consumer=observer['consumer'], observer_rebuilt=False,
        runtime_files={p.relative_to(bundle).as_posix():pin(p) for mode in ['runtime-control','runtime-observed'] for p in (bundle/mode).iterdir()})
    write(bundle/'built.json', built)
    review = dict(passed=True, built=pin(bundle/'built.json'), core_unchanged=True,
        consumer_unchanged=True, constructor_unchanged=True, **{k:v for k,v in observer.items() if k not in ['passed','inputs']})
    write(bundle/'build-review.json', review)
    shutil.copy2(bundle/'build-review.json', BASE/'build-review.json')
    manifest = read(APP/'collected/manifests/current-parakeet.json'); external = {}
    for value in manifest['models'].values(): external[value['path']] = {k:value[k] for k in ['bytes','sha256']}
    for value in [manifest['reference'], *[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP+'/assets/'+value['path']] = {k:value[k] for k in ['bytes','sha256']}
    for name in ['manifests/current-parakeet.json','runtime/protocol.py','runtime/campaign_processes.py']:
        path = APP/'collected'/name
        external[REMOTE_APP+'/'+name] = pin(path)
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    spec = dict(boot=1789634288.0, app=REMOTE_APP, external=external,
        source_receipt=pin(QUALIFIED_ROOT/'bundle/evidence/root-applied.json'),
        measured_source_receipt=pin(SOURCE/'prepared.json'), reference_product=observer['reference_product'],
        qualification_closures={k:pin(v/'closed.json') for k,v in QUALIFICATION.items()},
        diagnostic_context=dict(isolated_candidate=False, release_admitted=True, root_product_changed=True,
            application_admitted=True, graph_admitted=True, graph_controls_passed=True,
            failed_graph_cases=[], observer_rebuilt=False, retained_observer_review=observer['original_review'],
            actual_root_binaries=True, root_metadata_delta_verified=True, data_compiled_scope_equal=True),
        original_consumer=observer['consumer'], core=observer['product']['Lokad.Onnx.dll'], data=observer['product']['Lokad.Onnx.Data.dll'],
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json', spec)
    # Retain the original worker and audit, plus the explicit reference-identity adaptation.
    for folder in [TOOLS, ORIGINAL]:
        for path in folder.iterdir():
            if path.is_file():
                if path.suffix == '.py': ast.parse(path.read_text(encoding='utf8'), str(path))
                inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    path = TOOLS.parent/'ort-diagnosis-amd/run.py'
    inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    value = dict(passed=True, archive=pin(BASE/'payload.tar.gz'), spec=pin(bundle/'spec.json'), inputs=inputs)
    write(BASE/'prepared.json', value)
    print(json.dumps(dict(passed=True, archive=value['archive'], spec=value['spec'], observer_rebuilt=False)))
