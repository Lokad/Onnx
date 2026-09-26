"""Bind unchanged numerical consumers through qualified root and Pad inventories."""
import importlib.util
from pathlib import Path
from protocol import pin,read

ROOT=Path(__file__).resolve().parents[3]
CURRENT=ROOT/'artifacts/parakeet-owned-batch-isolation-models-amd-20260925'
QUALIFIED=ROOT/'artifacts/parakeet-owned-batch-isolation-root-policy-amd-20260925'
BUILD=ROOT/'artifacts/parakeet-pad-current-build-amd-20260926'
SCREEN=ROOT/'artifacts/parakeet-pad-current-screen-amd-20260926'
MEMORY=ROOT/'artifacts/parakeet-pad-memory-diagnostic-amd-20260926'


def reconcile(root,pad,model_products,selected,candidate):
    assert root['inventory_complete'] and pad['inventory_complete']
    assert len(root['observations'])==len(pad['observations'])==2
    for first,second,name,count in zip(root['observations'],pad['observations'],
            ('Lokad.Onnx.dll','Lokad.Onnx.Data.dll'),(3281,697),strict=True):
        assert first['assembly']==second['assembly']==name
        assert first['before_sha256']==model_products[name]['sha256']
        assert first['after_sha256']==second['before_sha256']==selected[name]['sha256']
        assert second['after_sha256']==candidate[name]['sha256']
        assert first['methods']==first['unchanged_methods']==second['methods']==count
        assert not first['differences'] and not first['added'] and not first['removed'] and not first['candidate_methods']
        assert len(first['normalized_methods'])==count
        assert first['normalized_methods']==second['normalized_methods']
        assert first['method_flags_before']==first['method_flags_after']==second['method_flags_before']
        before=set(first['public_surface']);after=set(first['public_surface_after'])
        assert not before-after
        additions={'MEMBER Lokad.Onnx.ComputationalGraph Method Int32 PrepareOwnedMatMulWeights()',
            'MEMBER Lokad.Onnx.ComputationalGraph Method Int32 PrepareOwnedMatMulWeights() FLAGS Public, HideBySig'}
        assert after-before==(additions if name=='Lokad.Onnx.dll' else set())
        assert after==set(second['public_surface']) and second['public_surface_equal']
    source=ROOT/'tests/parakeet/pad-current-build-amd/checks.py'
    loader=importlib.util.spec_from_file_location('pad_build_inventory_checks',source)
    checker=importlib.util.module_from_spec(loader);loader.loader.exec_module(checker)
    reviewed=checker.inventory(pad,selected,candidate)
    return dict(passed=True,qualified_model_product=model_products,selected=selected,candidate=candidate,
        original_public_bindings_preserved=True,all_data_methods_exact=True,
        all_original_method_flags_preserved=True,underlying_methods_reconciled=3978,
        padding=reviewed,no_consumer_or_product_build=True,
        scope='Transitive compiled compatibility from the product qualified with these exact consumers; '
              'all old public bindings retained and all Data methods exact. Full model execution remains required.')


def review():
    inputs={};proofs={}
    for folder,digest in [(CURRENT,'1997eeb782df89975f4893820bc0fc246dec60d1839b2c63a86ad2bba1aecd9d'),
            (QUALIFIED,'c77ef606c76508144c5656bdbe48aeac8948ce28396c8bee645587d31609c475'),
            (BUILD,'1835c79eda505c056cf796702ca734b6e43c65e066b3e54bb566ac25885c4018'),
            (SCREEN,'5b8d7df749d600238dfc6e1c9ead50486b0027754656c57d0c553573ada1b17d'),
            (MEMORY,'51e714628218f2908ac46d0e8336b29003d1f7a266e424bd37ec946fa1c7589b')]:
        assert pin(folder/'closed.json')['sha256']==digest
        proof=read(folder/'closed.json');assert proof['passed'];proofs[folder]=proof
        inputs[(folder/'closed.json').relative_to(ROOT).as_posix()]=pin(folder/'closed.json')
        for name in ['analysis.json','collected/collection.json','collected/inventory/instructions.json']:
            if name in proof['files']:
                assert pin(folder/name)==proof['files'][name],name
                inputs[(folder/name).relative_to(ROOT).as_posix()]=pin(folder/name)
    assert not proofs[SCREEN]['admitted'] and not proofs[MEMORY]['admitted']
    old=read(CURRENT/'analysis.json');root=read(QUALIFIED/'analysis.json');build=read(BUILD/'analysis.json')
    result=reconcile(read(QUALIFIED/'collected/inventory/instructions.json'),
        read(BUILD/'collected/inventory/instructions.json'),old['identities']['candidate'],root['built'],build['built'])
    for role,folder,products in [('selected',QUALIFIED/'collected/runtime',root['built']),
                                ('candidate',BUILD/'collected/runtime',build['built'])]:
        for name,wanted in products.items():
            assert pin(folder/name)==wanted
            inputs[(folder/name).relative_to(ROOT).as_posix()]=wanted
    consumers=old['consumers'];assert set(consumers)=={'AudioBenchmark','TranscribeReplay'}
    for name,wanted in consumers.items():
        path=CURRENT/'collected/runtimes/candidate'/(name+'.dll')
        assert pin(path)==wanted==proofs[CURRENT]['files'][path.relative_to(CURRENT).as_posix()]
        inputs[path.relative_to(ROOT).as_posix()]=wanted
    screen=read(SCREEN/'analysis.json')
    assert all(r['passed'] for r in screen['rows']) and len(screen['rows'])==12
    assert len([c for c in screen['controls'] if not c['passed']])==6
    assert read(MEMORY/'analysis.json')['memory']['passed']
    return dict(**result,consumers=consumers,inputs=inputs,component_screen_admitted=False,
        failed_component_controls=[c for c in screen['controls'] if not c['passed']],
        test_eligibility='PLAN decision after closed memory diagnosis; no component score changed.')
