"""Independently verify compiled scope and original/observed clock agreement."""
import json
from il_normalization import normalized_body


def compiled_scope(value, spec, built):
    assert value['inventory_complete'] and len(value['observations'])==1
    row, = value['observations']
    assert row['assembly']=='ReleaseBenchmark.dll' and row['public_surface_equal'] and row['compiler_rename'] is None
    assert row['before_sha256']==spec['previous_consumer']['sha256'] and row['after_sha256']==built['consumer']['sha256']
    assert not row['removed'] and row['unchanged_methods']==row['methods']-1
    key, = row['differences']; assert key.startswith('Program::<Main>$::')
    assert all(k.startswith(('ClockProbe','MatrixEvents')) for k in row['added'])
    assert all(row['method_flags_after'][k]==v for k,v in row['method_flags_before'].items())
    before = json.loads(row['normalized_methods'][key]); after = json.loads(row['candidate_methods'][key])
    rows = after['instructions']; removed = set()
    hooks = [
        [('ldloc.2',''),('ldloc.0',''),('ldfld','Program+<>c__DisplayClass0_0::System.String key'),('ldloc.3',''),
         ('call','ClockProbe::Void Initialize(System.String, System.String, System.String)')],
        [('ldloc.s','1D'),('call','ClockProbe::Void Begin(Int32)')],
        [('ldloc.s','1D'),('ldloc.s','1E'),('ldloc.s','20'),('call','ClockProbe::Void End(Int32, Int64, Int64)')],
        [('ldloc.2',''),('call','ClockProbe::Void Save(System.String)')],
    ]
    for expected in hooks:
        end, = [i for i,r in enumerate(rows) if (r['opcode'],r['operand'])==expected[-1]]
        start = end-len(expected)+1
        assert [(r['opcode'],r['operand']) for r in rows[start:end+1]]==expected
        removed.update(range(start,end+1))
    assert normalized_body(before)==normalized_body(after,[i for i in range(len(rows)) if i not in removed]), \
        'Original consumer instructions, branches, locals, stack or exception regions changed'
    return dict(passed=True,methods=row['methods'],unchanged_methods=row['unchanged_methods'],
        hooks=4,added_methods=len(row['added']),branches_locals_exceptions_equal=True,implementation_flags_equal=True)


def merge_observation(value, diagnostic):
    assert value['passed'] and value['mode']=='timing' and value['key'] in ['e5-8tok','e5-512tok']
    assert diagnostic['diagnosticOnly'] and value['pid']==diagnostic['pid']
    assert value['calls']==len(value['clocks'])==len(diagnostic['clocks'])==780
    clocks=[]
    for i,(clock,observed) in enumerate(zip(value['clocks'],diagnostic['clocks'],strict=True)):
        assert clock['index']==observed['index']==i and clock['warmup']==(i<600)
        assert clock['ticks']==observed['end']-observed['start']>0
        assert type(clock['frequency']) is int and clock['frequency']>0
        assert observed['marker']<=observed['start']<observed['end']
        clocks.append(dict(clock,**{k:v for k,v in observed.items() if k!='index'}))
    return dict(value,diagnosticOnly=True,nativeThread=diagnostic['nativeThread'],clocks=clocks)
