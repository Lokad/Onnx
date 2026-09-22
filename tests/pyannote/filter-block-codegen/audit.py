"""Close numerical/resource proof and retain all emitted code tiers without timing claims."""
import importlib.util
import json
import re
from run import ROOT, BASE, prepared
from protocol import JOBS, LIMITS, check_sample, pin, read, save
from checks import check_result


def listings(path):
    text=path.read_text(encoding='utf8')
    starts=list(re.finditer(r'; Assembly listing for method (.+) \(([^\n]+)\)\n',text))
    result=[]
    for index,start in enumerate(starts):
        end=starts[index+1].start() if index+1<len(starts) else len(text)
        body=text[start.start():end];sizes=re.findall(r'; Total bytes of code (\d+)',body)
        blocks=list(re.finditer(r'^(G_M\d+_IG\d+):[^\n]*\n',body,re.M));reductions=[]
        for i,block in enumerate(blocks):
            part=body[block.start():blocks[i+1].start() if i+1<len(blocks) else len(body)]
            fmas=len(re.findall(r'\bvfmadd\d*ps\b',part))
            if fmas:
                stack=[line for line in part.splitlines() if re.search(r'\b[xyz]mm(?:word)?\b|\b[xyz]mm\d+',line)
                       and re.search(r'\[(?:rbp|rsp)(?:[+\-\]])',line)]
                reductions.append(dict(label=block.group(1),fma_instructions=fmas,
                    input_broadcasts=len(re.findall(r'\bvbroadcastss\b',part)),vector_stack_references=stack,body=part))
        result.append(dict(method=start.group(1),tier=start.group(2),code_bytes=[int(x) for x in sizes],
            complete_uninterleaved=len(sizes)==1,line=text.count('\n',0,start.start())+1,reductions=reductions,body=body))
    return result


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    payload=read(BASE/'payload/payload.json');collected=BASE/'collected';receipt=read(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    transfer=read(BASE/'collection-transfer.json')
    assert transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    for name,wanted in payload['files'].items():assert pin(BASE/'payload'/name)==wanted,name
    module_path=ROOT/'tests/parakeet/portable-models/common.py'
    loader=importlib.util.spec_from_file_location('filter_codegen_resources',module_path)
    common=importlib.util.module_from_spec(loader);loader.loader.exec_module(common)
    common.verify(read(BASE/'inputs.json')['files'])
    local=common.resources(BASE,'controller.json',{'restore':(8,8,900,False),'build':(8,8,900,False),
        'local-production':(12,8,900,True),'local-candidate':(12,8,900,True)})
    local_state=read(BASE/'controller.json')
    for role in JOBS:
        result=read(BASE/'output'/('local-'+role+'.json'));check_result(result,role,payload,BASE/'payload',8)
        row,=[r for r in local_state['runs'] if r['name']=='local-'+role]
        assert result['runtime']=='10.0.12' and result['pid']==row['worker']['pid']
    state=read(collected/'identity.json');assert state['complete'] and state['code']==0
    assert state['supervisor']==read(BASE/'deployment.json') and state['boot_time']==1789634288.0
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[];reports={};code={}
    for row in state['runs']:
        role=row['name'];assert row['complete'] and row['code']==0 and row['seconds']<LIMITS['seconds']
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        assert row['preflight']==row['preflight_observations'][-1]==read(collected/(role+'-preflight.json'))[-1]
        samples=[json.loads(s) for s in (collected/'logs'/(role+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            for member in sample['members']:assert row['members'][str(member['pid'])]==member['birth']
        result=read(collected/role/'result.json');assert result['pid']==row['child']['pid'] and result['runtime']=='10.0.8'
        reports[role]=check_result(result,role,payload,BASE/'payload',16)
        code[role]=listings(collected/'logs'/(role+'.stdout'))
        required=['Kernel512']+(['Kernel512Four'] if role=='candidate' else [])
        for name in required:
            choices=[r for r in code[role] if r['method'].startswith('Lokad.Onnx.ConvBlockedSpatial:'+name+'(')
                     and r['tier'].startswith('Tier1') and r['complete_uninterleaved']]
            assert choices and any(r['reductions'] for r in choices),(role,name,'No complete optimized listing')
        resources.append(dict(name=role,samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started']<4*3600
    save(BASE/'listings.json',code)
    summaries={role:[{k:r[k] for k in ['method','tier','code_bytes','complete_uninterleaved','line']} |
                    dict(reductions=[{k:b[k] for k in ['label','fma_instructions','input_broadcasts','vector_stack_references']} for b in r['reductions']])
                    for r in rows] for role,rows in code.items()}
    analysis=dict(passed=True,reports=reports,resources=resources,local_resources=local['resources'],code=summaries,
        no_performance_measurement=True,tiering_and_isa_unchanged=True)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json',dict(passed=True,files=files,local_inputs=spec['files'],local_identities=local['identities'],remote_terminal=receipt['identities']))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),reports=reports,resources=resources,code=summaries)))


if __name__=='__main__':main()
