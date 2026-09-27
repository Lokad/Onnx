"""Bind the closed observer compatibility proof to the actual qualified root."""
import ast
import json
import shutil
import tarfile
from run import ROOT, TOOLS, ORIGINAL, PARENT, BASE, APP, REMOTE_APP, QUALIFIED_ROOT, native, pin, read, write

OLD = ROOT/'artifacts/parakeet-packed-final-row-profile-amd-20260925'
REVIEW = ROOT/'artifacts/parakeet-pointwise-tail-profile-review-20260927'
SOURCE = ROOT/'artifacts/parakeet-pointwise-tail-source-20260927/prepared.json'
QUALIFICATION = dict(app=APP, root=QUALIFIED_ROOT,
    pyannote=ROOT/'artifacts/parakeet-pointwise-tail-pyannote-app-amd-20260927',
    graphs=ROOT/'artifacts/parakeet-pointwise-tail-graphs-amd-20260927')


def retained_observer():
    assert pin(REVIEW/'closed.json')['sha256'] == 'e234eec85d6041f804b64fcec58d1978c8eed4e64dd56420bc0a65d25d351f99'
    closure = read(REVIEW/'closed.json')
    assert closure['passed'] and closure['read_only'] and closure['analysis'] == pin(REVIEW/'analysis.json')
    value = read(REVIEW/'analysis.json')
    assert value['passed'] and not value['observer_rebuilt'] and value['inference_calls'] == 0
    assert value['data_methods_exact'] == 697 and value['data_method_flags_equal']
    assert value['public_surfaces_and_assembly_attributes_equal']
    for name, wanted in value['inputs'].items():
        assert pin(ROOT/name) == wanted, name
    assert value['consumer']['sha256'] == '38ab5c7e65aa00ace50e8704348830286047e0c96ebc0a04b66c6e54d063899c'
    assert value['observed_data']['sha256'] == '51b6d2bb72640a99d8ed4e97334525dde8be60ac1ecb3c25ea690956bfcab2a2'
    return value


def diagnostic_gates():
    context = native.qualification()  # Refuses until the actual root digest is bound.
    observer = retained_observer()
    assert observer['candidate'] == context['measured']
    root = read(QUALIFIED_ROOT/'analysis.json')
    applied = read(QUALIFIED_ROOT/'bundle/evidence/root-applied.json')
    for label, key in [('pyannote','pyannote-app'), ('graphs',None)]:
        folder = QUALIFICATION[label]
        expected = applied['graph_qualification'] if key is None else applied['prerequisites'][key]
        assert pin(folder/'closed.json') == expected
        proof = read(folder/'closed.json')
        assert proof['passed'] and proof['admitted']
        for name, wanted in proof['files'].items():
            assert pin(folder/name) == wanted, name
    assert read(QUALIFICATION['pyannote']/'analysis.json')['identities']['candidate'] == context['measured']
    assert read(QUALIFICATION['graphs']/'analysis.json')['products']['candidate']['Lokad.Onnx.dll'] == context['measured']['Lokad.Onnx.dll']
    assert applied['prepared'] == pin(SOURCE)
    assert pin(SOURCE)['sha256'] == '9be6d0381e417b435b030b467cabd1d723398ffb9fac294ef8ff91c9b6a6e64c'
    return dict(passed=True, product=root['built'], reference_product=context['measured'],
        consumer=observer['consumer'], observed_data=observer['observed_data'],
        original_review=observer['original_observer_review'], observer_rebuilt=False,
        compatibility_review=pin(REVIEW/'closed.json'), root_closure=pin(QUALIFIED_ROOT/'closed.json'),
        methods=observer['preserved_observer_methods'], data_compiled_scope_equal=True,
        inputs={**observer['inputs'], **context['inputs']})


def prepare():
    assert not BASE.exists()
    observer = diagnostic_gates()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir()
    inputs = dict(observer['inputs'])
    def copy(path, name):
        target = bundle/name; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target); inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    copy(ORIGINAL/'remote.py', 'remote.py')
    copy(TOOLS/'README.md', 'prospective-plan.md')
    copy(APP/'collected/evidence/candidate-public.json', 'evidence/candidate-public.json')
    for label, folder in QUALIFICATION.items():
        for name in ['closed.json','analysis.json']:
            copy(folder/name, f'evidence/qualification/{label}/{name}')
    for name in ['closed.json','analysis.json']:
        copy(REVIEW/name, 'evidence/observer-compatibility/'+name)
    for name in ['build-review.json','build-collected/built.json','build-collected/inventory/instructions.json']:
        copy(OLD/name, 'evidence/retained-observer/'+name)
    copy(SOURCE, 'evidence/product-source.json')
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
        consumer_unchanged=True, constructor_unchanged=True,
        **{k:v for k,v in observer.items() if k not in ['passed','inputs']})
    write(bundle/'build-review.json', review)
    shutil.copy2(bundle/'build-review.json', BASE/'build-review.json')
    manifest = read(APP/'collected/manifests/current-parakeet.json'); external = {}
    for value in manifest['models'].values():
        external[value['path']] = {k:value[k] for k in ['bytes','sha256']}
    for value in [manifest['reference'], *[c['pcm'] for c in manifest['cases']]]:
        external[REMOTE_APP+'/assets/'+value['path']] = {k:value[k] for k in ['bytes','sha256']}
    for name in ['manifests/current-parakeet.json','runtime/protocol.py','runtime/campaign_processes.py']:
        path = APP/'collected'/name
        external[REMOTE_APP+'/'+name] = pin(path)
        inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    spec = dict(boot=1789634288.0, app=REMOTE_APP, external=external,
        source_receipt=pin(QUALIFIED_ROOT/'bundle/evidence/root-applied.json'),
        measured_source_receipt=pin(SOURCE), reference_product=observer['reference_product'],
        qualification_closures={k:pin(v/'closed.json') for k,v in QUALIFICATION.items()},
        diagnostic_context=dict(isolated_candidate=False, release_admitted=True, root_product_changed=True,
            application_admitted=True, graph_admitted=True, graph_controls_passed=True,
            failed_graph_cases=[], observer_rebuilt=False, retained_observer_review=observer['original_review'],
            observer_compatibility_review=observer['compatibility_review'],
            actual_root_binaries=True, root_metadata_delta_verified=True, data_compiled_scope_equal=True),
        original_consumer=observer['consumer'], core=observer['product']['Lokad.Onnx.dll'], data=observer['product']['Lokad.Onnx.Data.dll'],
        capture_limits=dict(available_before=11*1024**3,tmpfs_before=2*1024**3,rss=12*1024**3,seconds=900),
        minimum_free=1024**3,output_limit=512*1024**2,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    write(bundle/'spec.json', spec)
    for folder in [TOOLS, ORIGINAL, PARENT]:
        for path in folder.iterdir():
            if path.is_file():
                if path.suffix == '.py': ast.parse(path.read_text(encoding='utf8'), str(path))
                inputs[path.relative_to(ROOT).as_posix()] = pin(path)
    with tarfile.open(BASE/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(bundle.rglob('*')):
            if path.is_file(): archive.add(path, arcname=path.relative_to(bundle).as_posix(), recursive=False)
    value = dict(passed=True, archive=pin(BASE/'payload.tar.gz'), spec=pin(bundle/'spec.json'), inputs=inputs)
    write(BASE/'prepared.json', value)
    print(json.dumps(dict(passed=True, archive=value['archive'], spec=value['spec'], observer_rebuilt=False)))
