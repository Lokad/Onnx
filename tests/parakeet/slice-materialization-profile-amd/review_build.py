"""Admit byte-exact Core/Data products with one consumer identity-check edit."""
import base64
import importlib.util
import json
from run import BASE, TOOLS, PRIOR, PRELUDE, pin, read, write, ssh

loader=importlib.util.spec_from_file_location('layout_resources',TOOLS.parent/'slice-materialization-amd/review_build.py')
resource_checks=importlib.util.module_from_spec(loader);loader.loader.exec_module(resource_checks)
resources=resource_checks.resources


def main():
    resource_rows=resources('build');folder=BASE/'build-collected'
    transfer=read(BASE/'build-transfer.json')
    assert transfer['passed'] and transfer['collection']==pin(folder/'build-collection.json')
    assert transfer['archive']==pin(BASE/'build-results.tar.gz')
    spec=read(BASE/'bundle/spec.json');built=read(folder/'built.json')
    for key in ['core','candidate_core','data']:assert built[key]==spec[key]
    for name,wanted in built['runtime_files'].items():assert pin(folder/name)==wanted,name
    for runtime,core in [('runtime-control','core'),('runtime-observed','candidate_core')]:
        assert pin(folder/runtime/'Lokad.Onnx.dll')==spec[core]
        assert pin(folder/runtime/'Lokad.Onnx.Data.dll')==spec['data']
        assert pin(folder/runtime/'SampledAudio.dll')==built['consumer']
    for source in (PRIOR/'bundle/consumer-source').iterdir():
        if not source.is_file():continue
        actual=(BASE/'bundle/consumer-source'/source.name).read_bytes()
        if source.name=='Program.cs':
            replacement=b'==Environment.GetEnvironmentVariable("PARAKEET_PHASE_CORE_SHA")'
            assert actual.count(replacement)==1
            actual=actual.replace(replacement,b'=="'+spec['core']['sha256'].encode()+b'"')
        assert actual==source.read_bytes(),source.name
    inventory=read(folder/'inventory/instructions.json');assert inventory['inventory_complete']
    row,=inventory['observations']
    assert row['assembly']=='SampledAudio.dll' and row['public_surface_equal']
    assert row['before_sha256']==spec['original_consumer']['sha256']
    assert row['after_sha256']==built['consumer']['sha256']
    assert row['methods']==164 and row['unchanged_methods']==163 and not row['added'] and not row['removed']
    assert row['method_flags_before']==row['method_flags_after']
    key,=row['differences'];assert key.startswith('Program::<Main>$::')
    value=dict(passed=True,built=pin(folder/'built.json'),inventory=pin(folder/'inventory/instructions.json'),
        collection=pin(folder/'build-collection.json'),spec=pin(BASE/'bundle/spec.json'),
        model_closure=spec['model_closure'],products_byte_exact=True,
        consumer=dict(original_methods=164,unchanged=163,changed=key,added=0,removed=0,
            public_surface_equal=True,flags_equal=True,source_edit='explicit expected Core environment identity'),
        resources=resource_rows,reviewer=pin(__file__))
    write(BASE/'build-review.json',value)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    remote=ssh(PRELUDE+f'''
import base64
from remote import verify,pin,read,live
verify();state=read(base/'build-state.json');assert state['complete'] and state['code']==0
assert not live(state['supervisor']) and all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={value['built']!r} and pin(base/'inventory/instructions.json')=={value['inventory']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert remote['review']==pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',remote)
    print(json.dumps(dict(passed=True,review=remote['review'],consumer=value['consumer'])))


if __name__=='__main__':main()
