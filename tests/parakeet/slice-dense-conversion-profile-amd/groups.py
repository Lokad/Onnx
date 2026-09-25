"""Compare the prospectively defined executed positional group without double counting."""
import importlib.util
from pathlib import Path
import onnx


def compare_positional(selected,candidate,definition,model):
    phases=dict(selected=selected,candidate=candidate)
    rows={k:{(n['graph'],n['name']):n for n in p['node_rows']} for k,p in phases.items()}
    assert rows['selected'].keys()==rows['candidate'].keys()
    for key,a in rows['selected'].items():
        b=rows['candidate'][key]
        assert {k:v for k,v in a.items() if k not in ['ticks','corpus_seconds']}=={
            k:v for k,v in b.items() if k not in ['ticks','corpus_seconds']},key
    encoder={k:{name:r for (g,name),r in ns.items() if g=='encoder'} for k,ns in rows.items()}
    assert all(len(ns)==2856 for ns in encoder.values())
    path=Path(__file__).resolve().parents[1]/'slice-dense-conversion-results/define_groups.py'
    loader=importlib.util.spec_from_file_location('prospective_groups',path)
    module=importlib.util.module_from_spec(loader);loader.loader.exec_module(module)
    graph=onnx.load(model,load_external_data=False).graph
    expected=definition['groups']['managed'];assert len(expected['nodes'])==34
    for nodes in encoder.values():assert module.scope(graph,nodes)==expected
    targets=set(expected['targets']);assert len(targets)==24
    union=[]
    for node in expected['nodes']:
        name=node['name'];a,b=(encoder[k][name] for k in ['selected','candidate'])
        assert a['calls']==b['calls']==60
        before,after=a['corpus_seconds'],b['corpus_seconds']
        union.append(dict(**node,selected_seconds=before,candidate_seconds=after,
            gain=None if before==0 else 1-after/before,projection=name in targets))
    before=sum(r['selected_seconds'] for r in union);after=sum(r['candidate_seconds'] for r in union)
    kernels=[r for r in union if r['projection']]
    all_nodes=[]
    for (g,name),a in rows['selected'].items():
        b=rows['candidate'][(g,name)]
        all_nodes.append(dict(graph=g,name=name,op=a['op'],calls=a['calls'],
            selected_seconds=a['corpus_seconds'],candidate_seconds=b['corpus_seconds'],
            difference_seconds=a['corpus_seconds']-b['corpus_seconds'],
            in_positional_group=g=='encoder' and name in {n['name'] for n in union}))
    return dict(passed=after<before and all(r['candidate_seconds']<r['selected_seconds'] for r in kernels),
        selected_seconds=before,candidate_seconds=after,complete_group_gain=1-after/before,
        selected_kernel_seconds=sum(r['selected_seconds'] for r in kernels),
        candidate_kernel_seconds=sum(r['candidate_seconds'] for r in kernels),
        improved_kernels=sum(r['candidate_seconds']<r['selected_seconds'] for r in kernels),
        kernels=24,nodes=34,shared_ancestors_counted_once=True,rows=union,all_nodes=all_nodes)
