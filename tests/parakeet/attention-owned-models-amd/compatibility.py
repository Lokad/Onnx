"""Reconcile the current root and one-method candidate with admitted model bytes."""
from pathlib import Path
from protocol import pin,read

ROOT=Path(__file__).resolve().parents[3]
CURRENT=ROOT/'artifacts/parakeet-pointwise-tail-models-amd-20260927'
QUALIFIED=ROOT/'artifacts/parakeet-pointwise-tail-root-amd-20260927'
CONTRACTS=ROOT/'artifacts/parakeet-attention-owned-build-amd-20260928'
CENSUS=ROOT/'artifacts/parakeet-attention-owned-census-amd-20260928'
PROOFS=[
    (CURRENT,'models','closed.json','f773277daa848121c199c58e67bce3777677c2ee2654fbe26250d02719155960'),
    (QUALIFIED,'root','closed.json','fc11676361a50ef613f783fcb51e4ead488c9c2ee34a3edddcbb561f67037e47'),
    (CONTRACTS,'contracts','closed.json','a293403a5dc3a7bdabb7fdc448470feea2b54dbf139d98b7b3f9ed0eef39854f'),
    (CENSUS,'census','closed.json','ef4e41eca49d33f6cac4ae7d510922053874002787709cc3d83bc98c3437d3d7')]


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
        changed={'Lokad.Onnx.ComputationalGraph::PrepareOwnedMatMulWeights'} if name=='Lokad.Onnx.dll' else set()
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
    census=read(CENSUS/'analysis.json')
    build=read(CONTRACTS/'build-review.json')
    selected,candidate=root['built'],contracts['product']
    assert contracts['compiled_review']==pin(CONTRACTS/'build-review.json')
    assert build['passed'] and build['arithmetic_leaves_unchanged'] and build['product']==candidate
    assert [(r['mode'],r['passed'],r['skipped']) for r in contracts['suites']]==[('normal',93,0),('256',93,0),('scalar',26,0)]
    assert census['passed'] and census['product']==candidate and census['contracts']==pin(CONTRACTS/'closed.json')
    assert census['original_compiled_review']==pin(CONTRACTS/'build-review.json')
    assert [(r['mode'],r['result']['owned_count'],r['result']['retained_maps']) for r in census['modes']]==[('512',179,37),('256',179,37)]
    inventories=[QUALIFIED/'collected/inventory/instructions.json',CONTRACTS/'build-collected/logs/instructions.json']
    scope=reconcile(*map(read,inventories),model['identities']['candidate'],selected,candidate)
    assert scope==build['methods']
    consumers=model['consumers'];assert set(consumers)=={'TranscribeReplay','AudioBenchmark'}
    for name,wanted in consumers.items():
        path=CURRENT/'collected/runtimes/candidate'/(name+'.dll')
        assert pin(path)==wanted;inputs[path.relative_to(ROOT).as_posix()]=wanted
    current_runtime=ROOT/'artifacts/parakeet-pointwise-tail-profile-amd-20260927/bundle/runtime-control'
    for directory,product in [(current_runtime,selected),(CONTRACTS/'build-collected/runtime',candidate)]:
        for name,wanted in product.items():
            path=directory/name;assert pin(path)==wanted;inputs[path.relative_to(ROOT).as_posix()]=wanted
    for path in [*inventories,CONTRACTS/'build-review.json']:
        inputs[path.relative_to(ROOT).as_posix()]=pin(path)
    original={n.removeprefix('source/'):v for n,v in read(QUALIFIED/'bundle/stage.json')['files'].items() if n.startswith('source/')}
    assert len(original)==445
    for name,wanted in original.items():assert pin(ROOT/name)==wanted,name
    inputs.update(original)
    return dict(passed=True,selected=selected,candidate=candidate,consumers=consumers,inputs=inputs,
        selected_runtime=read(CONTRACTS/'bundle/spec.json')['prior'],compiled_scope=scope,
        qualified_model_product=model['identities']['candidate'],underlying_methods_reconciled=3985,
        original_public_bindings_preserved=True,all_data_methods_exact=True,all_original_method_flags_preserved=True,
        no_consumer_or_product_build=True,no_performance_claim=True,focused_contracts=pin(CONTRACTS/'closed.json'),
        actual_model_census=pin(CENSUS/'closed.json'),test_eligibility='Exact scope and ownership qualified; complete numerical/public corpus remains required.')
