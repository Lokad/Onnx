"""Audit the complete AMD arithmetic and generated-code evidence independently."""
from pathlib import Path
import argparse,importlib.util,json,math
import numpy as np
from common import CORE,FILTER,LIMITS,REMOTE,pin,read,write,verify
from code_audit import inspect

def original():
    path=Path(__file__).with_name('original_audit.py')
    if not path.exists():path=Path(__file__).resolve().parent.parent/'layernorm-output/audit.py'
    spec=importlib.util.spec_from_file_location('layernorm_original_audit',path);value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value

def identity(value,probe,phase):
    flags={} if phase=='proof' else dict(COMPlus_JitDisasm=FILTER,COMPlus_JitStdOutFile=REMOTE+'/result/code/jit.txt')
    assert value['mode']==phase and value['core_sha256']==CORE and value['probe_sha256']==probe['sha256']
    assert value['runtime']=='10.0.8' and value['affinity']==4 and value['vector_width']==8
    assert value['vector512_hardware'] is True and value['avx512'] is True and value['settings']==flags

def resources(state,rows):
    assert state['code']==0 and state['terminal_members'] is True and 0<state['seconds']<LIMITS['seconds']
    assert len(rows)==state['samples']>0 and state['started']<state['ended']
    assert rows[0]['seconds']<5 and state['seconds']-rows[-1]['seconds']<5
    assert all(0<b['seconds']-a['seconds']<5 for a,b in zip(rows,rows[1:]))
    observed={};peak=0
    for row in rows:
        assert 0<=row['seconds']<=state['seconds'] and row['available']>=LIMITS['available']
        assert len({m['pid'] for m in row['members']})==len(row['members'])
        for m in row['members']:
            assert m['affinity']==[2] and m['rss']>=0 and m['birth']>=state['child']['birth']
            key=str(m['pid']);assert state['members'][key]==m['birth']
            assert key not in observed or observed[key]==m['birth'];observed[key]=m['birth']
        peak=max(peak,sum(m['rss'] for m in row['members']))
    assert observed==state['members'] and observed[str(state['child']['pid'])]==state['child']['birth']
    assert peak==state['peak_rss']<=LIMITS['rss']
    return dict(samples=len(rows),peak_rss=peak,minimum_available=min(r['available'] for r in rows),births=[dict(pid=int(pid),birth=birth) for pid,birth in observed.items()])

def audit(base):
    meta=verify(base);old=original();capture=read(base/'capture/capture.json');probe=pin(base/'bin/LayerNormAmdProof.dll')
    state=read(base/'result/identity.json');assert state['complete'] is True and state['code']==0 and not state.get('error')
    assert state['bundle']==pin(base/'bundle.json') and state['limits']==LIMITS and [r['phase'] for r in state['runs']]==['proof','code']
    assert state['runs'][0]['ended']<=state['runs'][1]['started']
    cpu=(base/'result/cpuinfo.txt').read_text();assert 'AuthenticAMD' in cpu and 'avx512f' in cpu
    assert capture['passed'] is True and capture['identity']['core_sha256']==CORE and [r['name'] for r in capture['cases']]==old.CASES
    old.inventory(base/'capture',capture,'capture.json');nodes=[]
    for case in capture['cases']:
        folder=base/'capture'/case['name'];assert case==read(folder/'capture.json') and len(case['nodes'])==25
        for index,node in enumerate(case['nodes']):
            assert node['index']==index and node['block']==384 and node['product_replay_bitwise'] is True
            prefix=f'{index:02}';width=node['block'];rows=node['outer'];count=width*rows
            x=old.array(folder/(prefix+'-x.f32'),count).reshape(rows,width);scale=old.array(folder/(prefix+'-scale.f32'),width)
            bias=old.array(folder/(prefix+'-bias.f32'),width) if node['has_bias'] else None
            y=old.array(folder/(prefix+'-y.f32'),count).reshape(rows,width);reference=old.reference(x,scale,bias,node['epsilon'])
            nodes.append(dict(case=case['name'],index=index,prefix=case['name']+'-'+prefix,node=node,x=x,scale=scale,bias=bias,y=y,reference=reference))
    assert len(nodes)==125;phases={};births=[state['supervisor']]
    for run in state['runs']:
        phase=run['phase'];folder=base/'result'/phase;output=folder/'output';proof=read(output/'proof.json')
        identity(proof['identity'],probe,phase);assert run['flags']==proof['identity']['settings']
        old.inventory(output,proof,'proof.json')
        assert proof['passed'] is True and proof['cases']==len(proof['records'])==915 and proof['captured']==125
        assert len({r['name'] for r in proof['records']})==915
        assert proof['comparisons']==sum(r['values']*4 for r in proof['records'])==32182096
        assert proof['capture_sha256']==pin(base/'capture/capture.json')['sha256'] and old.synthetic(proof['records'])==790
        for r in proof['records']:
            assert all(r[k] is True for k in ['exact','guards','inputs_preserved','inplace'])
            assert math.isfinite(r['scalar_error']) and 0<=r['scalar_error']<=1e-5
        assert proof['maximum_scalar_error']==max(r['scalar_error'] for r in proof['records'])
        compared=0;maximum=0
        for item,record in zip(nodes,proof['records'][790:]):
            node=item['node'];assert record['name']==item['case']+':'+str(item['index']) and record['prefix']==item['prefix']
            assert record['block']==node['block'] and record['outer']==node['outer'] and record['values']==item['y'].size
            assert record['has_bias']==node['has_bias'] and np.float32(record['epsilon'])==np.float32(node['epsilon'])
            assert record['input_sha256']==old.sha(item['x']) and record['scale_sha256']==old.sha(item['scale'])
            assert record['bias_sha256']==(None if item['bias'] is None else old.sha(item['bias']))
            for variant in ['copy','wide']:
                path=output/(item['prefix']+'-'+variant+'.f32');actual=old.array(path,item['y'].size).reshape(item['y'].shape)
                assert old.sha(actual)==old.sha(item['y'])==record['product_sha256']
                error=old.scaled(actual,item['reference']);assert error<=1e-5;maximum=max(maximum,error);compared+=actual.size
        assert compared==15475200
        telemetry=resources(run,[json.loads(line) for line in (folder/'samples.jsonl').read_text().splitlines()]);births+=telemetry['births']
        phases[phase]=dict(cases=915,comparisons=32182096,independent_values=compared,maximum_scalar_error=maximum,resources=telemetry)
    warm=read(base/'result/code/warmup.json');identity(warm['identity'],probe,'code')
    node=next(n for n in nodes if n['case']=='e5-30tok' and n['index']==0)
    assert warm['passed'] is True and type(warm['pairs']) is int and warm['pairs']>0 and 3<=warm['seconds']<LIMITS['seconds']
    assert warm['block']==384 and warm['outer']==30 and np.float32(warm['epsilon'])==np.float32(node['node']['epsilon'])
    assert warm['input_sha256']==old.sha(node['x']) and warm['scale_sha256']==old.sha(node['scale'])
    assert warm['bias_sha256']==(None if node['bias'] is None else old.sha(node['bias']))
    assert warm['output_sha256']==old.sha(node['y'])
    code=inspect((base/'result/code/jit.txt').read_text())
    return dict(passed=True,bundle=pin(base/'bundle.json'),phases=phases,code=code,warmup=warm,births=births,
                scope='Actual AMD arithmetic and generated-code qualification only; no kernel/model/ORT speed claim')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--payload',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    value=audit(args.payload.resolve());write(args.output,value);print(json.dumps({k:v for k,v in value.items() if k not in ['births','warmup']},indent=2))
