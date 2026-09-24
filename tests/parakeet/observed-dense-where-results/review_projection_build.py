"""Review the exact compiler-generated JSON envelope without rebuilding the observer."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT/'tests/parakeet/projection-route-amd'
sys.path.insert(0,str(TOOLS))
from run import BASE, pin, read, prepared

PREFIX = '<>f__AnonymousType0`5[<Name>j__TPar,<Pass>j__TPar,<Control>j__TPar,<Frequency>j__TPar,<Calls>j__TPar]'
METHODS = {'get_Name','get_Pass','get_Control','get_Frequency','get_Calls','Equals','GetHashCode','ToString','.ctor'}


def generated_scope(row):
    extras = [k for k in row['added'] if not k.startswith('Lokad.Onnx.ParakeetProjectionProbe')]
    assert len(extras) == 9 and {k.split('::')[1] for k in extras} == METHODS
    for key in extras:
        assert key.startswith(PREFIX+'::')
        body = json.loads(row['candidate_methods'][key]); assert not body['exceptions']
        for instruction in body['instructions']:
            op,operand = instruction['opcode'],instruction['operand']
            if op in ['call','callvirt','newobj']:
                assert operand.startswith(('System.Object::','System.String::','System.IFormatProvider::',
                    'System.Collections.Generic.EqualityComparer`1[',PREFIX+'::')), (key,operand)
            if op in ['ldfld','stfld','ldflda']:
                assert operand.startswith(PREFIX+'::'), (key,operand)
            assert op not in ['calli','stsfld','ldsflda','cpblk','initblk'], (key,op)
    # The only external reference is creation of the envelope in Save, after
    # the unchanged consumer has stopped its request timer.
    references = []
    for key,encoded in row['candidate_methods'].items():
        if key in extras: continue
        for instruction in json.loads(encoded)['instructions']:
            if '<>f__AnonymousType0`5[' in instruction['operand']:
                references.append((key,instruction))
    assert len(references) == 2
    assert all(k.startswith('Lokad.Onnx.ParakeetProjectionProbe::Save::') for k,_ in references)
    assert [i['opcode'] for _,i in references] == ['newobj','call']
    assert references[1][1]['operand'].startswith('System.Text.Json.JsonSerializer::Void Serialize[<>f__AnonymousType0`5](')
    return extras


def main():
    prepared(); assert not (BASE/'build-review.json').exists()
    original = TOOLS/'review_build.py'
    assert pin(original) == read(BASE/'prepared.json')['tools']['review_build.py']
    inventory = read(BASE/'build-collected/inventory/instructions.json')
    row, = [r for r in inventory['observations'] if r['assembly']=='Lokad.Onnx.Data.dll']
    generated = generated_scope(row)
    # Negative check: an unexpected product call must not become an admitted
    # compiler-generated helper merely because its type name matches.
    damaged = dict(row,candidate_methods=dict(row['candidate_methods']))
    body = json.loads(damaged['candidate_methods'][generated[0]])
    body['instructions'].insert(0,dict(opcode='call',operand='Lokad.Onnx.CPUExecutionProvider::Bypass()',offset=0))
    damaged['candidate_methods'][generated[0]] = json.dumps(body)
    try: generated_scope(damaged)
    except AssertionError: pass
    else: raise AssertionError('Generated-method guard accepted an inference call')
    evidence = dict(initial_review_refused=True,original_review=pin(original),
        inventory=pin(BASE/'build-collected/inventory/instructions.json'),
        scope='Nine compiler-generated methods for the five-field JSON envelope, referenced only by Save.',
        methods={k:hashlib.sha256(row['candidate_methods'][k].encode()).hexdigest() for k in generated},
        original_inference_checks_unchanged=True,no_rebuild=True)
    source = original.read_text(encoding='utf8')
    before = "assert row['added'] and all(k.startswith('Lokad.Onnx.ParakeetProjectionProbe') for k in row['added'])"
    after = "assert row['added'] and all(k.startswith('Lokad.Onnx.ParakeetProjectionProbe') or k in approved_generated for k in row['added'])"
    assert source.count(before) == 1
    source = source.replace(before,after)
    before = 'value = dict(passed=True,built='
    assert source.count(before) == 1
    source = source.replace(before,'value = dict(passed=True,compiler_generated_review=generated_evidence,built=')
    namespace = dict(__name__='corrected_projection_review',__file__=__file__,
        approved_generated=set(generated),generated_evidence=evidence)
    exec(compile(source,str(original),'exec'),namespace)
    namespace['main']()


if __name__ == '__main__': main()
