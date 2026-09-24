"""Require exact original methods and the sole inherited conversion override."""
import base64
import json
import re
from run import BASE, ORIGINAL, PRELUDE, pin, read, write, ssh, prepared
from checks import resources


def main():
    prepared();assert not (BASE/'build-review.json').exists()
    original_resources=resources('build',original=True)
    resource_rows=resources('build');folder=BASE/'build-collected';built=read(folder/'built.json')
    assert [r['name'] for r in resource_rows]==['bridge-restore','bridge-build','inventory']
    for name,wanted in built['runtime_files'].items():assert pin(folder/name)==wanted,name
    inventory=read(folder/'inventory/instructions.json');assert inventory['inventory_complete']
    row,=inventory['observations'];spec=read(BASE/'bundle/spec.json')
    assert row['assembly']=='Lokad.Onnx.dll' and row['before_sha256']==spec['core']['sha256']
    assert row['after_sha256']==built['core']['sha256']
    assert not row['removed'] and not row['differences'] and row['methods']==row['unchanged_methods']==3253
    for key,flags in row['method_flags_before'].items():assert row['method_flags_after'][key]==flags,key
    added,=row['added'];assert added=='Lokad.Onnx.TensorSlice`1[T]::ToDenseTensor::Lokad.Onnx.DenseTensor`1[T] ToDenseTensor()'
    assert row['effective_conversion_signature_equal'] and not row['public_surface_equal'] and row['removed_surface']==[]
    prefix='MEMBER Lokad.Onnx.TensorSlice`1 Method Lokad.Onnx.DenseTensor`1[T] ToDenseTensor()'
    assert set(row['added_surface'])=={prefix,prefix+' FLAGS Public, Virtual, HideBySig'}
    assert set(row['public_surface_after'])==set(row['public_surface'])|set(row['added_surface'])
    body=json.loads(row['candidate_methods'][added]);assert body['exceptions']==[]
    assert body['locals']==[dict(type='Lokad.Onnx.DenseTensor`1[T]',IsPinned=False)]
    instructions=body['instructions']
    calls=[r['operand'] for r in instructions if r['opcode'] in ['call','callvirt']]
    assert calls==['Lokad.Onnx.TensorSlice`1[T]::Boolean TryCopyContiguousSlice(Lokad.Onnx.DenseTensor`1[T] ByRef)',
        'Lokad.Onnx.Tensor`1[T]::Lokad.Onnx.DenseTensor`1[T] ToDenseTensor()']
    assert [r['opcode'] for r in instructions]==['ldarg.0','ldloca.s','call','brfalse.s','ldloc.0','ret','ldarg.0','call','ret']
    assert instructions[1]['operand']=='00'
    target=instructions[3]['offset']+2+int.from_bytes(bytes.fromhex(instructions[3]['operand']),'little',signed=True)
    assert target==instructions[6]['offset']
    warnings=[]
    for path in (ORIGINAL/'build-collected/logs').glob('*.stdout'):
        for line in path.read_text(encoding='utf8').splitlines():
            if ': warning ' in line:
                assert path.name=='tensors-build.stdout' and 'Zzz.WideProjectionEntry.cs(20,' in line and 'warning CS8604:' in line
                assert "parameter 'x'" in line or "parameter 'y'" in line
                warnings.append(line)
    assert len(warnings)==4
    assert not any(': warning ' in p.read_text(encoding='utf8') for p in (folder/'logs').glob('*.stdout'))
    assert not built['product_rebuilt']
    assert built['core']==pin(ORIGINAL/'build-collected/runtime/Lokad.Onnx.dll')
    assert built['consumer']==pin(ORIGINAL/'build-collected/runtime/Lokad.Onnx.Tensors.Tests.dll')
    value=dict(passed=True,built=pin(folder/'built.json'),inventory=pin(folder/'inventory/instructions.json'),
        collection=pin(folder/'build-collection.json'),spec=pin(BASE/'bundle/spec.json'),
        methods=dict(original=3253,unchanged=3253,added=added,original_flags_equal=True,
            declared_surface_equal=False,effective_conversion_signature_equal=True,only_declared_addition=row['added_surface']),
        existing_warnings=warnings,no_new_warning=True,resources=resource_rows,original_resources=original_resources,
        original_failure_preserved=pin(ORIGINAL/'build-collected/build-collection.json'),product_rebuilt=False,reviewer=pin(__file__))
    write(BASE/'build-review.json',value)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    result=ssh(PRELUDE+f'''
import base64
from remote import verify,read,pin,live,idle
verify();idle();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={value['built']!r} and pin(base/'inventory/instructions.json')=={value['inventory']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert result['review']==pin(BASE/'build-review.json');write(BASE/'build-review-transferred.json',result)
    print(json.dumps(dict(passed=True,review=result['review'],core=built['core'],methods=value['methods'])))


if __name__=='__main__':main()
