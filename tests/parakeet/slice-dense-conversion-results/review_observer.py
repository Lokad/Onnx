"""Extend the existing light observer's compiled proof to ordinary current Data."""
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
TOOLS=ROOT/'tests/parakeet/observed-dense-where-profile-amd'
loader=importlib.util.spec_from_file_location('retained_profile_scope',TOOLS/'run.py')
original=importlib.util.module_from_spec(loader);loader.loader.exec_module(original)
pin,read=original.pin,original.read


def main():
    output=OUT/'observer-20260925.json';assert not output.exists()
    scope=original.observer_scope();assert scope['passed'] and scope['no_rebuild']
    inventory=ROOT/'artifacts/parakeet-observed-dense-where-inventory-amd-20260924/collected/inventory/instructions.json'
    old_inventory=original.PRIOR/'build-collected/inventory/instructions.json'
    old=next(r for r in read(old_inventory)['observations'] if r['assembly']=='Lokad.Onnx.Data.dll')
    current=next(r for r in read(inventory)['observations'] if r['assembly']=='Lokad.Onnx.Data.dll')
    assert current['methods']==current['unchanged_methods']==697
    assert not current['added'] and not current['removed'] and not current['differences']
    assert current['public_surface_equal'] and current['public_surface']==old['public_surface']
    assert current['normalized_methods']==old['normalized_methods']
    assert current['method_flags_before']==current['method_flags_after']==old['method_flags_before']
    models=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
    assert pin(models/'closed.json')['sha256']=='5860fc366c3d976907f35f8d7a1832c10566fa08910af32c168dca7a63979842'
    proof=read(models/'closed.json');analysis=read(models/'analysis.json')
    assert proof['passed'] and analysis['passed'] and proof['analysis']==pin(models/'analysis.json')
    data=analysis['identities']['selected']['Lokad.Onnx.Data.dll']
    assert data==analysis['identities']['candidate']['Lokad.Onnx.Data.dll']
    assert current['after_sha256']==data['sha256']=='a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
    graph=ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924/capture-collected/wall/graphs.json'
    assert read(graph)['decoder_joint-model.onnx']['retained_packed_bytes']==51461120
    result=dict(passed=True,no_rebuild=True,no_inference=True,original_scope=scope,current_data=data,
        original_data_methods=697,all_original_bodies_flags_and_surface_exact=True,
        current_inventory=pin(inventory),original_inventory=pin(old_inventory),current_model_closure=pin(models/'closed.json'),
        graph_metadata=pin(graph),decoder_packed_bytes=51461120,
        observer_runtime=original.OBSERVER.relative_to(ROOT).as_posix()+'/build-collected/runtime-observed',
        consumer=scope['consumer'],observed_data=scope['data'],reviewer=pin(Path(__file__)))
    with output.open('x',encoding='utf8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(passed=True,review=pin(output),consumer=scope['consumer'],data=scope['data'],methods=697)))


if __name__=='__main__':main()
