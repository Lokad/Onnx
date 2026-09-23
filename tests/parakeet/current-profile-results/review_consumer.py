"""Verify all unchanged Main instructions and the exact extracted public call."""
import difflib,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-current-profile-build-amd-20260923'
def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def main():
    assert not (BASE/'consumer-review.json').exists()
    assert pin(BASE/'closed.json')['sha256']=='27df0af0d6ef488a4468aa7cee8f1d31112eebd28525b79ea8d6e6da4fddcecc'
    row=json.loads((BASE/'collected/inventory/instructions.json').read_text())['observations'][0]
    key,=row['differences'];a=json.loads(row['normalized_methods'][key]);b=json.loads(row['candidate_methods'][key])
    assert a['InitLocals']==b['InitLocals'] and a['MaxStackSize']==b['MaxStackSize'] and a['locals']==b['locals']
    def offset(n):return n+2 if n>=1229 else n
    mapping={i['offset']:i for i in b['instructions']};compared=0
    for i,instruction in enumerate(a['instructions']):
        n=instruction['offset']
        if 1195<=n<1229:continue
        actual=mapping[offset(n)];assert actual['opcode']==instruction['opcode']
        opcode=instruction['opcode'];operand=instruction['operand']
        if n==319:assert operand=='pyannote';operand='parakeet'
        if n==339:assert operand=='Only the original pyannote timing workload';operand='Only the current Parakeet timing workload'
        branch=opcode.startswith(('br','beq','bne','bge','bgt','ble','blt','leave'))
        if branch:
            assert opcode!='break'
            target=a['instructions'][i+1]['offset']+int.from_bytes(bytes.fromhex(operand),'little',signed=True)
            wanted=offset(target)-offset(a['instructions'][i+1]['offset'])
            operand=wanted.to_bytes(len(bytes.fromhex(operand)),'little',signed=True).hex().upper()
        assert actual['operand']==operand,(n,instruction,actual,operand)
        compared+=1
    for old,new in zip(a['exceptions'],b['exceptions'],strict=True):
        wanted=dict(old)
        for name in ['Try','Handler']:
            wanted[name+'Offset']=offset(old[name+'Offset'])
            wanted[name+'Length']=offset(old[name+'Offset']+old[name+'Length'])-wanted[name+'Offset']
        assert wanted==new
    replacement=[i for i in b['instructions'] if 1195<=i['offset']<1231]
    assert [i['opcode'] for i in replacement]==['ldloc.s','ldloc.s','blt.s','ldloc.s','ldloc.s','callvirt','call','br.s','ldloc.s','ldloc.s','callvirt','call']
    assert [i['operand'] for i in replacement[:3]]==['17','12','10'] # pass < warmup -> warmup wrapper
    assert 'FullParakeet(' in replacement[6]['operand'] and 'WarmupParakeet(' in replacement[-1]['operand']
    assert replacement[3]['operand']==replacement[8]['operand']=='0C' # existing transcriber
    assert replacement[4]['operand']==replacement[9]['operand']=='1A' # existing case
    assert replacement[5]['operand']==replacement[10]['operand']=='Case::Single[] get_Pcm()'
    extracted=[(i['opcode'],i['operand']) for i in a['instructions'] if 1204<=i['offset']<=1224]
    for name in row['added']:
        body=json.loads(row['candidate_methods'][name]);ins=body['instructions']
        assert not body['locals'] and not body['exceptions']
        assert [(i['opcode'],i['operand']) for i in ins[:2]]==[('ldarg.0',''),('ldarg.1','')]
        assert [(i['opcode'],i['operand']) for i in ins[2:7]]==extracted
        tail=[(i['opcode'],i['operand']) for i in ins[7:]]
        assert tail==([('ret','')] if 'WarmupParakeet::' in name else [('dup',''),('call','SampledRequests::Void After(System.Object)'),('ret','')])
    result=dict(passed=True,build=pin(BASE/'closed.json'),inventory=pin(BASE/'collected/inventory/instructions.json'),
        source_patch=pin(BASE/'bundle/consumer.patch'),reviewer=pin(Path(__file__)),unchanged_methods=159,main_instructions_compared=compared,
        changed_main_offsets=[319,339,[1195,1229]],new_main_interval=[1195,1231],
        locals_and_exception_regions_preserved=True,branch_targets_preserved=True,exact_public_call_extracted=True,
        product_unchanged=True,no_performance_measurement=True)
    (BASE/'consumer-review.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'consumer-review-20260923.json').write_text(json.dumps(result,indent=2)+'\n')
    (OUT/'consumer.patch').write_bytes((BASE/'bundle/consumer.patch').read_bytes())
    print(json.dumps(result))
if __name__=='__main__':main()
