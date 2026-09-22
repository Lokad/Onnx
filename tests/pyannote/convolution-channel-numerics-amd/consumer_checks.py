"""Constrain all consumer methods, including exact rebasing for wider channel literals."""
from copy import deepcopy
import json

OLD = '3ca0a2a5ff1b129272da05b679d539d865a3c6c20ee8625c24955e2a5f497c6a'
CHANNELS={'channels64':(64,80),'channels128':(128,256)}


def expected_channels(before,channels):
    """Compute exact expected IL; every branch/exception endpoint is retained."""
    original=json.loads(before);body=deepcopy(original)
    rows=original['instructions'];changes={422:(16,channels[0]),427:(32,channels[1])}
    increments={offset:0 if value<=127 else 3 for offset,(_,value) in changes.items()}
    def shift(offset):return offset+sum(delta for point,delta in increments.items() if point<offset)
    branch_stems={'br','brfalse','brtrue','beq','bge','bgt','ble','blt','bne','leave'}
    for index,(old,row) in enumerate(zip(rows,body['instructions'],strict=True)):
        offset=old['offset'];row['offset']=shift(offset)
        if offset in changes:
            previous,value=changes[offset]
            assert old['opcode']=='ldc.i4.s' and old['operand']==previous.to_bytes(1,'little').hex().upper()
            assert rows[index-1]['opcode']==('ldc.i4.0' if offset==422 else 'ldc.i4.1') and rows[index+1]['opcode']=='stelem.i4'
            row['opcode']='ldc.i4.s' if value<=127 else 'ldc.i4'
            row['operand']=value.to_bytes(1 if value<=127 else 4,'little',signed=True).hex().upper()
        elif old['opcode'].split('.')[0] in branch_stems:
            data=bytes.fromhex(old['operand']);assert len(data) in [1,4]
            following=rows[index+1]['offset'];target=following+int.from_bytes(data,'little',signed=True)
            assert target in {r['offset'] for r in rows}
            row['operand']=(shift(target)-shift(following)).to_bytes(len(data),'little',signed=True).hex().upper()
        else:assert old['opcode']!='switch'
    assert len([r for r in rows if r['offset'] in changes])==2
    for old,row in zip(original['exceptions'],body['exceptions'],strict=True):
        for prefix in ['Try','Handler']:
            start=old[prefix+'Offset'];end=start+old[prefix+'Length']
            row[prefix+'Offset']=shift(start);row[prefix+'Length']=shift(end)-shift(start)
        if old['filter']>=0:row['filter']=shift(old['filter'])
    return body


def consumer_inventory(value, mode, previous, current, core):
    assert value['inventory_complete']
    row, = value['observations']
    assembly = 'LayerGraphs.dll' if mode == 'layers' else 'Lokad.Onnx.Backend.Tests.dll'
    assert row['assembly'] == assembly and row['public_surface_equal']
    assert not row['added'] and not row['removed'] and row.get('compiler_rename') is None
    count = 134 if mode == 'layers' else 145
    assert row['methods'] == len(row['normalized_methods']) == count and row['unchanged_methods'] == count-1
    key, = row['differences']
    assert key == ('ModelProbe' if mode == 'layers' else 'Probe')+'::Main::Int32 Main(System.String[])'
    assert set(row['candidate_methods']) == {key}
    before, after = row['normalized_methods'][key], row['candidate_methods'][key]
    assert before.count(OLD) == 2
    if mode in CHANNELS:
        assert expected_channels(before.replace(OLD,core),CHANNELS[mode])==json.loads(after)
    else:assert before.replace(OLD,core)==after
    assert row['before_sha256'] == previous['sha256'] and row['after_sha256'] == current['sha256']
    return dict(passed=True,mode=mode,methods=count,unchanged=count-1,changed=[key],
        only_core_literal_changed=mode not in CHANNELS,channel_literals=list(CHANNELS[mode]) if mode in CHANNELS else None,all_branch_exception_targets_preserved=True)
