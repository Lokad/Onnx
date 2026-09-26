"""Preserve Pyannote qualification while passing both expected assembly hashes."""
import copy
import json

OLD_DATA = 'a893952f583f680ad9dcf677a32b9393541814396a35c6a4eb18a1e7325cbae1'
OLD_USAGE = 'root manifest new-output expected-core-sha'
NEW_USAGE = OLD_USAGE+' expected-data-sha'


def source(original):
    changes = [
        ('if (args.Length != 4) throw new ArgumentException("'+OLD_USAGE+'");',
         'if (args.Length != 5) throw new ArgumentException("'+NEW_USAGE+'");'),
        ('Require(Sha(typeof(Community1Diarizer).Assembly.Location) == "'+OLD_DATA+'", "Qualified data");',
         'Require(Sha(typeof(Community1Diarizer).Assembly.Location) == args[4], "Qualified data");')]
    result=original
    for before,after in changes:
        assert result.count(before.encode())==1
        result=result.replace(before.encode(),after.encode())
    return result


def expected_main(original):
    """Derive all expected IL, including offsets, branches and exception ranges."""
    value=copy.deepcopy(original)
    instructions=original['instructions']
    matches=[i for i,r in enumerate(instructions) if r['opcode']=='ldstr' and r['operand']==OLD_DATA]
    index,=matches
    start=instructions[index]['offset'];end=instructions[index+1]['offset']
    assert end-start==5
    def moved(position):
        assert not start<position<end, 'A control-flow boundary enters the replaced literal'
        return position-2 if position>=end else position
    result=[]
    branches={'br','br.s','brfalse','brfalse.s','brtrue','brtrue.s','beq','beq.s',
        'bge','bge.s','bgt','bgt.s','ble','ble.s','blt','blt.s','bne.un','bne.un.s',
        'bge.un','bge.un.s','bgt.un','bgt.un.s','ble.un','ble.un.s','blt.un','blt.un.s','leave','leave.s'}
    for i,old in enumerate(instructions):
        if i==index:
            result.extend(dict(offset=start+j,opcode=opcode,operand='') for j,opcode in
                          enumerate(['ldarg.0','ldc.i4.4','ldelem.ref']))
            continue
        row=copy.deepcopy(old);row['offset']=moved(old['offset'])
        assert row['opcode']!='switch', 'Review a switch table explicitly if the consumer gains one'
        if row['opcode'] in branches:
            raw=bytes.fromhex(row['operand']);assert len(raw) in [1,4]
            next_offset=instructions[i+1]['offset']
            target=next_offset+int.from_bytes(raw,'little',signed=True)
            distance=moved(target)-moved(next_offset)
            row['operand']=distance.to_bytes(len(raw),'little',signed=True).hex().upper()
        result.append(row)
    assert result[5]==dict(offset=9,opcode='ldc.i4.4',operand='')
    result[5]['opcode']='ldc.i4.5'
    usages=[r for r in result if r['opcode']=='ldstr' and r['operand']==OLD_USAGE]
    usage,=usages;usage['operand']=NEW_USAGE
    value['instructions']=result
    for region in value['exceptions']:
        for key,length in [('TryOffset','TryLength'),('HandlerOffset','HandlerLength')]:
            begin,finish=region[key],region[key]+region[length]
            region[key],region[length]=moved(begin),moved(finish)-moved(begin)
        if region['filter']>=0:region['filter']=moved(region['filter'])
    return value


def inventory(value, before, after):
    assert value['inventory_complete'] and len(value['observations'])==1
    row=value['observations'][0]
    assert row['assembly']=='GraphQualification.dll'
    assert row['before_sha256']==before['sha256'] and row['after_sha256']==after['sha256']
    assert row['methods']==96 and row['unchanged_methods']==95
    assert row['public_surface_equal'] and row['compiler_rename'] is None
    assert not row['removed'] and not row['added']
    key,=row['differences'];assert key.startswith('Program::<Main>$::')
    original=json.loads(row['normalized_methods'][key]);actual=json.loads(row['candidate_methods'][key])
    assert actual==expected_main(original), 'Changed more than argument count, usage or the expected-Data operand'
    return dict(passed=True,methods=96,unchanged=95,main_changes=['argument-count','usage','expected-data-argument'],
        assembly_hash_comparison_preserved=True,all_other_instructions_locals_branches_exceptions_exact=True)
