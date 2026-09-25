"""Require the original Data body and the already reviewed observer helpers."""
import json


def verify_data(row,before_sha256,after_sha256,reference):
    assert row['assembly']==reference['assembly']=='Lokad.Onnx.Data.dll'
    assert row['before_sha256']==before_sha256 and row['after_sha256']==after_sha256
    assert row['public_surface_equal'] and not row['removed']
    assert (row['methods'],row['unchanged_methods'])==(697,696)
    assert len(row['normalized_methods'])==697
    assert len(row['method_flags_before'])==697
    assert set(row['method_flags_before'])==set(row['normalized_methods'])
    for key,flags in row['method_flags_before'].items():assert row['method_flags_after'][key]==flags,key
    key,=row['differences']
    assert key.startswith('Lokad.Onnx.ParakeetTranscriber::Execute::')
    # Every other original method, including the constructor that enables
    # packed weights, must match the measured M78 baseline in the inventory.
    assert len(row['added'])==50 and set(row['added'])==set(reference['added'])
    assert set(row['candidate_methods'])==set(row['added'])|{key}
    assert set(row['method_flags_after'])==set(row['method_flags_before'])|set(row['added'])
    for added in row['added']:
        assert row['candidate_methods'][added]==reference['candidate_methods'][added],added
        assert row['method_flags_after'][added]==reference['method_flags_after'][added],added
    before=json.loads(row['normalized_methods'][key]);after=json.loads(row['candidate_methods'][key])
    assert before['InitLocals']==after['InitLocals'] and before['MaxStackSize']==after['MaxStackSize']
    body=[dict(i,offset=i['offset']-7) for i in after['instructions'] if 7<=i['offset']<120]
    assert body==before['instructions'][:-1],'Original Execute instructions or branches changed'
    assert [i['opcode'] for i in after['instructions'][:3]]==['ldarg.0','call','stloc.0']
    assert after['instructions'][1]['operand']=='Lokad.Onnx.ParakeetPhaseProbe::Scope Enter(Lokad.Onnx.GraphExecution)'
    assert [i['opcode'] for i in after['instructions'] if i['offset']>=120]==[
        'stloc.1','leave.s','ldloca.s','constrained.','callvirt','endfinally','ldloc.1','ret']
    assert after['exceptions']==[dict(flags=2,TryOffset=7,TryLength=116,HandlerOffset=123,HandlerLength=14,filter=-1,caught=None)]
    # The entire hooked method must retain the existing reviewed observer's
    # locals, operand identities and terminal disposal instructions as well.
    assert row['normalized_methods'][key]==reference['normalized_methods'][key]
    assert row['candidate_methods'][key]==reference['candidate_methods'][key]
    return dict(passed=True,original_methods=697,unchanged=696,changed=key,
        added_helpers=50,original_body_recovered=True,observer_helpers_exact=True,
        constructor_unchanged=True,original_flags_equal=True,public_surface_equal=True)


def verify_runner(row,digest):
    assert row['assembly']=='SampledAudio.dll'
    assert row['before_sha256']==row['after_sha256']==digest
    assert row['public_surface_equal'] and not row['removed'] and not row['added'] and not row['differences']
    assert row['methods']==row['unchanged_methods']==len(row['normalized_methods'])==164
    assert not row['candidate_methods']
    assert row['method_flags_before']==row['method_flags_after'] and len(row['method_flags_before'])==164
    return dict(passed=True,methods=164,unchanged=164,byte_identical=True,flags_equal=True,public_surface_equal=True)
