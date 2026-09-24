"""Qualify seven retained cases plus the diagnosed e5 successor; preserve both campaigns."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
ORDER=['current-a','candidate-a','ort-a','ort-b','candidate-b','current-b']


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))

GRAPH=ROOT/'artifacts/parakeet-validated-composition-graphs-amd-20260924'
E5=ROOT/'artifacts/e5-warmed-qualification-amd-20260924'
BASE=ROOT/'artifacts/parakeet-composition-graph-qualification-20260924'
OUT=Path(__file__).resolve().parent


def verified(folder):
    proof=read(folder/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    return proof,read(folder/'analysis.json')


def statistics(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def derive():
    original,old=verified(GRAPH);successor,new=verified(E5)
    assert pin(GRAPH/'closed.json')['sha256']=='729814e9effff9c5e1456e165e7ef792bf2cba1c4978ce28ee3cb3ea7b4d1209'
    assert not original['admitted'] and not original['all_controls_passed']
    assert [r['key'] for r in old['performance'] if not r['qualified']]==['e5-30tok']
    assert old['clocks']==37512 and old['measured']==8640
    assert new['clocks']==8289 and new['measured']==1080 and len(new['setups'])==9
    assert new['original_graph_closure']==pin(GRAPH/'closed.json')
    assert new['diagnosis_closure']['sha256']=='8304dc71470c5666b83078f1fa487ae8595259d3ce9738471f70065d4919e990'
    assert old['passed'] and new['passed'] and not old['root_product_changed'] and not new['root_product_changed']
    for analysis in [old,new]:
        assert analysis['consumer']['implementation_flags_equal'] and analysis['consumer']['branches_locals_exceptions_equal']
    assert [(r['before'],r['after']) for r in new['consumer']['changes']]==[(780,1380),(600,1200)]
    products=read(GRAPH/'payload.json')['products'];assert products==read(E5/'payload.json')['products']==new['products']
    old_stats=statistics(ROOT/'tests/parakeet/validated-composition-graphs-amd/statistics.py','original_graph_statistics')
    new_stats=statistics(ROOT/'tests/benchmarks/e5-warmed-qualification-amd/statistics.py','diagnosed_e5_statistics')
    performance=[];clocks=[];setups=[]
    for row in old['performance']:
        key=row['key'];use_new=key=='e5-30tok';folder=E5 if use_new else GRAPH
        source='e5-successor' if use_new else 'original-graphs'
        stats=new_stats if use_new else old_stats
        values={role:read(folder/'collected'/f'timing-{key}-{role}'/'output/result.json') for role in ORDER}
        result=stats.summarize(values)
        expected=new['performance'] if use_new else row
        assert expected==dict(key=key,**result)
        performance.append(dict(key=key,source=source,warmups=1200 if use_new else 600,measured_per_process=180,**result))
        for name in [f'verify-{key}-{role}' for role in ['current','candidate','ort']]+[f'timing-{key}-{role}' for role in ORDER]:
            value=read(folder/'collected'/name/'output/result.json')
            assert value['passed'] and value['inputs_unchanged'] and value['held_outputs_unchanged'] and value['flags']=={}
            role=name.split('-')[-1] if name.startswith('verify-') else name.split('-')[-2]
            if role!='ort':assert value['core']==products[role]['Lokad.Onnx.dll']['sha256']
            assert all(0<=a['max_scaled_error']<=1e-4 for a in value['arrays'])
            clocks.extend(dict(source=source,process=name,**c) for c in value['clocks'])
            setups.append(dict(source=source,process=name,seconds=value['setup_seconds']))
    assert len(performance)==8 and len(clocks)==41112 and sum(not c['warmup'] for c in clocks)==8640 and len(setups)==72
    assert successor['admitted']==new['performance']['qualified']
    admitted=all(r['qualified'] for r in performance)
    result=dict(passed=True,admitted=admitted,all_controls_passed=all(c['passed'] for r in performance for c in r['controls']),
        products=products,performance=performance,clocks=len(clocks),measured=8640,setups=setups,
        source_closures=dict(original_graphs=pin(GRAPH/'closed.json'),e5_successor=pin(E5/'closed.json')),
        source_calls_retained=45801,original_graph_failure_preserved=True,root_product_changed=False,
        consumer=dict(original=old['consumer'],e5=new['consumer']))
    return result,clocks


def qualify():
    assert not BASE.exists();analysis,clocks=derive();BASE.mkdir()
    (BASE/'analysis.json').write_text(json.dumps(analysis,indent=2,allow_nan=False)+'\n',encoding='utf8')
    with (BASE/'clocks.csv').open('x',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(clocks[0]));writer.writeheader();writer.writerows(clocks)
    closure=dict(passed=True,admitted=analysis['admitted'],all_controls_passed=analysis['all_controls_passed'],
        source_closures=analysis['source_closures'],generator=pin(Path(__file__)),
        files={p.name:pin(p) for p in BASE.iterdir() if p.is_file()})
    (BASE/'closed.json').write_text(json.dumps(closure,indent=2)+'\n',encoding='utf8')
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),admitted=analysis['admitted'],clocks=analysis['clocks'])))


def admission():
    proof,value=verified(BASE);actual,_=derive()
    assert value==actual and proof['source_closures']==value['source_closures']
    assert proof['admitted']==value['admitted'] and proof['all_controls_passed']==value['all_controls_passed']
    assert proof['generator']==pin(Path(__file__))
    assert proof['admitted'] and proof['all_controls_passed']
    return value


if __name__=='__main__':qualify()
