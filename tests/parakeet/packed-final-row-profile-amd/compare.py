"""Match constant encoder projections through actual graph edges, including ORT's scale."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import onnx
from run import ROOT, BASE, pin, read, write


def main():
    closure=read(BASE/'closed.json');assert closure['passed'] and closure['analysis']==pin(BASE/'analysis.json')
    managed=read(BASE/'analysis.json')
    native_base=ROOT/'artifacts/parakeet-ort-diagnosis-amd-20260924'
    assert read(native_base/'closed.json')['passed']
    assert read(native_base/'closed.json')['analysis']==pin(native_base/'analysis.json')
    native=read(native_base/'analysis.json')
    graph_review=ROOT/'artifacts/parakeet-ort-graph-review-20260924'
    assert read(graph_review/'closed.json')['passed']
    assert read(graph_review/'closed.json')['analysis']==pin(graph_review/'analysis.json')
    reviewed=read(graph_review/'analysis.json')
    graph_path=ROOT/'artifacts/parakeet-ort-graphs-amd-v2-20260924/collected/encoder/optimized.onnx'
    assert pin(graph_path)==reviewed['graphs']['encoder']['model']
    graph=onnx.load(graph_path,load_external_data=False).graph
    ort_nodes={n.name:n for n in graph.node}
    nodes=[n for n in managed['phases']['wall']['node_rows'] if n['graph']=='encoder']
    by_name={n['name']:n for n in nodes};assert len(by_name)==len(nodes)
    producers={edge:n for n in nodes for edge in n['outputs'] if edge}
    consumers=defaultdict(list)
    for node in nodes:
        for edge in node['inputs']:
            if edge:consumers[edge].append(node)
    matched=[];used=set();groups=defaultdict(lambda:dict(nodes=0,managed=0.,ort=0.))
    for row in reviewed['projections']:
        if not row['constant_b']:continue
        name=row['name'].removesuffix('/MatMulScaleFusion/')
        assert name in by_name,name
        node=by_name[name];native_node=ort_nodes[row['name']]
        assert node['op']=='MatMul' and len(node['inputs'])==2
        assert node['inputs']==list(native_node.input)
        weight=node['constant_inputs'][1]
        assert weight is not None and weight['name']==row['b_name'] and weight['dims']==[row['k'],row['n']]
        members=[node];edge,=node['outputs'];target,=native_node.output
        while edge!=target:
            successor,=consumers[edge]
            assert successor['op'] in ['Mul','Div'],successor['name']
            assert len(successor['inputs'])==2 and edge in successor['inputs']
            members.append(successor)
            for other in successor['inputs']:
                if other==edge:continue
                constant=producers.get(other)
                if constant is not None:
                    assert constant['op']=='Constant',constant['name']
                    # Include private scale construction; shared constants remain in other work.
                    if len(consumers[other])==1:members.append(constant)
                else:assert any(c and c['name']==other for c in successor['constant_inputs'])
            edge,=successor['outputs']
            assert len(members)<=4
        assert row['alpha']==(.5 if len(members)>1 else 1.)
        for member in members:
            assert member['id'] not in used;used.add(member['id'])
            assert member['calls']==60
        seconds=sum(n['corpus_seconds'] for n in members)
        value=dict(ort_name=row['name'],shape=[row['k'],row['n']],alpha=row['alpha'],
            managed_nodes=[n['name'] for n in members],managed_seconds=seconds,ort_seconds=row['corpus_seconds'],
            excess_seconds=seconds-row['corpus_seconds'],ratio=seconds/row['corpus_seconds'])
        matched.append(value);key=(row['k'],row['n'],row['alpha']);group=groups[key]
        group['nodes']+=1;group['managed']+=seconds;group['ort']+=row['corpus_seconds']
    assert len(matched)==217 and len(used)==265
    grouped=[dict(reduction=k,columns=n,alpha=alpha,**v,excess=v['managed']-v['ort']) for (k,n,alpha),v in groups.items()]
    grouped.sort(key=lambda r:-r['excess'])
    total_managed=sum(r['managed_seconds'] for r in matched);total_ort=sum(r['ort_seconds'] for r in matched)
    comparison=dict(passed=True,managed_closure=pin(BASE/'closed.json'),native_closure=pin(native_base/'closed.json'),
        graph_review=pin(graph_review/'closed.json'),groups=grouped,projections=matched,
        managed_projection_seconds=total_managed,ort_projection_seconds=total_ort,
        projection_excess_seconds=total_managed-total_ort,managed_group_node_count=len(used),
        managed_phases=managed['phases']['phase']['phase_seconds'],ort_phases=native['phases']['control']['corpus_phase_seconds'],
        managed_complete=managed['corpus'],phase_over_control=managed['phase_over_control'],wall_over_phase=managed['wall_over_phase'],
        profiler_clocks_are_diagnostic=True,native_profile_is_historical=True,
        fresh_cross_engine_score=False,source_receipt=managed['source_receipt'])
    write(BASE/'comparison.json',comparison)
    out=ROOT/'tests/parakeet/packed-final-row-profile-results';out.mkdir(exist_ok=True)
    with (out/'projections-20260925.csv').open('x',newline='',encoding='utf8') as stream:
        fields=['ort_name','shape','alpha','managed_nodes','managed_seconds','ort_seconds','excess_seconds','ratio']
        writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(matched)
    with (out/'observations-20260925.json').open('x',encoding='utf8') as stream:
        json.dump({k:v for k,v in comparison.items() if k!='projections'},stream,indent=2);stream.write('\n')
    print(json.dumps({k:v for k,v in comparison.items() if k!='projections'}))


if __name__=='__main__':main()
