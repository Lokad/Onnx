"""Reconcile the current root and one-method transpose candidate with admitted model bytes."""
from pathlib import Path
from protocol import pin,read

ROOT=Path(__file__).resolve().parents[3]
CURRENT=ROOT/'artifacts/parakeet-attention-owned-models-amd-20260928'
QUALIFIED=ROOT/'artifacts/parakeet-attention-owned-root-recovery-amd-20260928'
CONTRACTS=ROOT/'artifacts/parakeet-transpose-axis-build-recovery-amd-20260928'
PROOFS=[
    (CURRENT,'models','closed.json','2d64fad91c26c00ffe7cd003aa128efc4bc92280da96a9bb805a8e2b93412177'),
    (QUALIFIED,'root','closed.json','ee4a38ff671cc3fd8cd608c0dc3008f5c1b99f61ba6c139f3deeaf0e9039305e'),
    (CONTRACTS,'contracts','closed.json','c0c063bb3f73eed2d8d7372d17dd1da3ce4f6a962e190ef101b504fb245e4ecd')]


def reconcile(root,candidate_inventory,model_product,selected,candidate):
    assert root['inventory_complete'] and candidate_inventory['inventory_complete']
    assert len(root['observations'])==len(candidate_inventory['observations'])==2
    scope=[]
    for first,second,name,count in zip(root['observations'],candidate_inventory['observations'],
            ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll'],[3288,697],strict=True):
        assert first['assembly']==second['assembly']==name
        assert first['before_sha256']==model_product[name]['sha256']
        assert first['after_sha256']==second['before_sha256']==selected[name]['sha256']
        assert second['after_sha256']==candidate[name]['sha256']
        assert first['methods']==first['unchanged_methods']==second['methods']==count
        assert not first['differences'] and not first['added'] and not first['removed'] and not first['candidate_methods']
        assert len(first['normalized_methods'])==count and first['normalized_methods']==second['normalized_methods']
        assert first['method_flags_before']==first['method_flags_after']==second['method_flags_before']==second['method_flags_after']
        assert not second['added'] and not second['removed']
        changed={'Lokad.Onnx.Tensor`1[T]::TransposeInto'} if name=='Lokad.Onnx.dll' else set()
        assert {'::'.join(n.split('::')[:2]) for n in second['differences']}==changed
        assert len(second['differences'])==len(changed) and second['unchanged_methods']==count-len(changed)
        for row in [first,second]:
            assert row['public_surface_equal'] and row['public_surface']==row['public_surface_after']
            assert row['assembly_attributes_before']==row['assembly_attributes_after']
        assert first['public_surface_after']==second['public_surface']
        assert first['assembly_attributes_after']==second['assembly_attributes_before']
        assert set(second['candidate_methods'])==set(second['differences'])
        scope.append(dict(assembly=name,original=count,unchanged=second['unchanged_methods'],changed=second['differences']))
    return scope


def review():
    inputs={}
    for folder,label,filename,digest in PROOFS:
        path=folder/filename;assert pin(path)['sha256']==digest
        proof=read(path);assert proof['passed'] and proof['analysis']==pin(folder/'analysis.json')
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,(label,name)
        inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    model=read(CURRENT/'analysis.json')
    root=read(QUALIFIED/'analysis.json')
    contracts=read(CONTRACTS/'analysis.json')
    build=read(CONTRACTS/'build-review.json')
    selected,candidate=root['built'],contracts['product']
    assert contracts['compiled_review']==pin(CONTRACTS/'build-review.json')
    assert build['passed'] and build['arithmetic_leaves_unchanged'] and build['product']==candidate
    expected=[(mode,suite,38 if suite=='backend' else 83 if mode=='scalar' else 84,0)
              for mode in ['normal','256','scalar','faces-off'] for suite in ['backend','tensors']]
    assert [(r['mode'],r['suite'],r['passed'],r['skipped']) for r in contracts['suites']]==expected
    inventories=[QUALIFIED/'collected/inventory/instructions.json',CONTRACTS/'build-collected/logs/instructions.json']
    scope=reconcile(*map(read,inventories),model['identities']['candidate'],selected,candidate)
    assert scope==build['methods']
    consumers=model['consumers'];assert set(consumers)=={'TranscribeReplay','AudioBenchmark'}
    for name,wanted in consumers.items():
        path=CURRENT/'collected/runtimes/candidate'/(name+'.dll')
        assert pin(path)==wanted;inputs[path.relative_to(ROOT).as_posix()]=wanted
    current_runtime=ROOT/'artifacts/parakeet-attention-owned-profile-amd-20260928/bundle/runtime-control'
    for directory,product in [(current_runtime,selected),(CONTRACTS/'build-collected/runtime',candidate)]:
        for name,wanted in product.items():
            path=directory/name;assert pin(path)==wanted;inputs[path.relative_to(ROOT).as_posix()]=wanted
    for path in [*inventories,CONTRACTS/'build-review.json']:
        inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    original={n.removeprefix('source/'):v for n,v in read(QUALIFIED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(original)==446
    for name,wanted in original.items():assert pin(ROOT/name)==wanted,name
    inputs.update(original)
    return dict(passed=True,selected=selected,candidate=candidate,consumers=consumers,inputs=inputs,
        selected_runtime=read(CONTRACTS/'bundle/spec.json')['prior'],compiled_scope=scope,
        qualified_model_product=model['identities']['candidate'],underlying_methods_reconciled=3985,
        original_public_bindings_preserved=True,all_data_methods_exact=True,all_original_method_flags_preserved=True,
        no_consumer_or_product_build=True,no_performance_claim=True,focused_contracts=pin(CONTRACTS/'closed.json'),
        test_eligibility='One transpose dispatch method and all 487 bit/layout/fallback contracts qualified; complete numerical/public corpus remains required.')
