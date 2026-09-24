"""Prove the combined Core preserves recurrence plus the exact qualified copy IL."""
import base64
import importlib.util
import json
from run import ROOT, BASE, PRIOR, PRELUDE, pin, read, write, ssh

RESOURCE_REVIEW=ROOT/'tests/parakeet/slice-materialization-amd/review_build.py'
loader=importlib.util.spec_from_file_location('layout_resource_review',RESOURCE_REVIEW)
shared=importlib.util.module_from_spec(loader);loader.loader.exec_module(shared)
resources=shared.resources


def main():
    resource_rows=resources('build');folder=BASE/'build-collected'
    built=read(folder/'built.json');spec=read(BASE/'bundle/spec.json')
    for name,wanted in built['runtime_files'].items():assert pin(folder/name)==wanted,name
    assert built['data']==spec['data']==pin(PRIOR/'collected/runtime/Lokad.Onnx.Data.dll')
    inventory=read(folder/'inventory/instructions.json');assert inventory['inventory_complete']
    row,=inventory['observations'];assert row['assembly']=='Lokad.Onnx.dll'
    assert row['before_sha256']==spec['core']['sha256']==pin(PRIOR/'collected/runtime/Lokad.Onnx.dll')['sha256']
    assert row['after_sha256']==built['core']['sha256']
    assert row['public_surface_equal'] and not row['removed'] and row['methods']==3250 and row['unchanged_methods']==3249
    for key,flags in row['method_flags_before'].items():assert row['method_flags_after'][key]==flags,key
    changed,=row['differences'];added,=row['added']
    assert changed.startswith('Lokad.Onnx.TensorSlice`1[T]::Reshape::')
    assert added.startswith('Lokad.Onnx.TensorSlice`1[T]::TryCopyContiguousSlice::')
    assert row['method_flags_after'][added]==0
    expected=read(BASE/'bundle/copy-methods.json')
    assert row['candidate_methods']==expected['methods'],'Qualified copy implementation changed'
    before=json.loads(row['normalized_methods'][changed]);after=json.loads(row['candidate_methods'][changed])
    tail=after['instructions'][-len(before['instructions']):];start=tail[0]['offset']
    assert [dict(i,offset=i['offset']-start) for i in tail]==before['instructions']
    value=dict(passed=True,built=pin(folder/'built.json'),inventory=pin(folder/'inventory/instructions.json'),
        collection=pin(folder/'build-collection.json'),spec=pin(BASE/'bundle/spec.json'),
        methods=dict(original=3250,unchanged=3249,changed=changed,added=added,original_flags_equal=True,
            public_surface_equal=True,original_fallback_exact=True,qualified_copy_bodies_exact=True),
        recurrence_core=spec['core'],data_unchanged=True,source_prepared=spec['source_prepared'],
        resources=resource_rows,reviewer=pin(__file__),resource_reviewer=pin(RESOURCE_REVIEW))
    write(BASE/'build-review.json',value)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    result=ssh(PRELUDE+f'''
import base64
from remote import verify,pin,read,live
verify();state=read(base/'build-state.json');assert state['complete'] and state['code']==0
assert not live(state['supervisor']) and all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={value['built']!r} and pin(base/'inventory/instructions.json')=={value['inventory']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert result['review']==pin(BASE/'build-review.json');write(BASE/'build-review-transferred.json',result)
    print(json.dumps(dict(passed=True,review=result['review'],core=built['core'],data=built['data'],methods=value['methods'])))


if __name__=='__main__':main()
