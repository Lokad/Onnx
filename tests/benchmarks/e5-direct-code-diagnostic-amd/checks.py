"""Review only the logging predicate and preserve the complete observed clocks."""
import json
from il_normalization import normalized_body
from protocol import CALLS


def compiled_scope(value,spec,built):
    assert value['inventory_complete'] and len(value['observations'])==1
    row,=value['observations']
    assert row['assembly']=='ReleaseBenchmark.dll' and row['public_surface_equal'] and row['compiler_rename'] is None
    assert row['before_sha256']==spec['previous_consumer']['sha256'] and row['after_sha256']==built['consumer']['sha256']
    assert row['methods']==131 and row['unchanged_methods']==130 and not row['removed']
    key,=row['differences'];assert key=='Program::<Main>$::Void <Main>$(System.String[])'
    added,=row['added'];assert added=='DisassemblyPolicy::Allowed::Boolean Allowed(System.Collections.Generic.Dictionary`2[System.String,System.String])'
    assert all(row['method_flags_after'][k]==v for k,v in row['method_flags_before'].items())
    assert row['method_flags_after'][added]==0
    before=json.loads(row['normalized_methods'][key]);after=json.loads(row['candidate_methods'][key])
    kept=[]
    for body,expected in [(before,[
        ('callvirt','System.Collections.Generic.Dictionary`2[System.String,System.String]::Int32 get_Count()'),
        ('ldc.i4.0',''),('ceq','')]),(after,[
        ('call','DisassemblyPolicy::Boolean Allowed(System.Collections.Generic.Dictionary`2[System.String,System.String])')])]:
        rows=body['instructions']
        at,=[i for i,r in enumerate(rows) if (r['opcode'],r['operand'])==('ldstr','Runtime override')]
        assert [(r['opcode'],r['operand']) for r in rows[at-len(expected):at]]==expected
        kept.append([i for i in range(len(rows)) if not at-len(expected)<=i<at])
    assert normalized_body(before,kept[0])==normalized_body(after,kept[1]),'Consumer changed outside logging predicate'
    return dict(passed=True,methods=131,unchanged_methods=130,changed_flag_predicates=1,added_methods=1,
        calls=CALLS,timing_numerical_ownership_observation_instructions_equal=True,
        branches_locals_exceptions_flags_equal=True)


def merge_observation(value,diagnostic):
    assert value['passed'] and value['mode']=='timing' and value['key']=='e5-8tok'
    assert diagnostic['diagnosticOnly'] and value['pid']==diagnostic['pid']
    assert value['calls']==len(value['clocks'])==len(diagnostic['clocks'])==CALLS
    clocks=[]
    for i,(clock,observed) in enumerate(zip(value['clocks'],diagnostic['clocks'],strict=True)):
        assert clock['index']==observed['index']==i and clock['warmup']==(i<600)
        assert clock['ticks']==observed['end']-observed['start']>0
        assert type(clock['frequency']) is int and clock['frequency']>0
        assert observed['marker']<=observed['start']<observed['end']
        clocks.append(dict(clock,**{k:v for k,v in observed.items() if k!='index'}))
    return dict(value,diagnosticOnly=True,nativeThread=diagnostic['nativeThread'],clocks=clocks)
