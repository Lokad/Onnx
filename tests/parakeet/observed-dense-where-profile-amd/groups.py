"""Reconcile all 72 Where kernels and their previously matched input boundaries."""
from collections import Counter
import importlib.util
from pathlib import Path
import sys
import onnx

ROOT=Path(__file__).resolve().parents[3]
HELPERS=ROOT/'tests/parakeet/managed-phase-amd'
sys.path.append(str(HELPERS))
loader=importlib.util.spec_from_file_location('masking_graphs',HELPERS/'compare_masking_padding.py')
matched_graphs=importlib.util.module_from_spec(loader);loader.loader.exec_module(matched_graphs)


def compare_where(selected,candidate,matched,model):
    original=onnx.load(model,load_external_data=False).graph
    phases={'selected':selected,'candidate':candidate}
    nodes={label:{n['name']:n for n in phase['node_rows'] if n['graph']=='encoder'} for label,phase in phases.items()}
    assert all(len(value)==2856 for value in nodes.values())
    assert nodes['selected'].keys()==nodes['candidate'].keys()
    for name,row in nodes['selected'].items():
        assert {k:v for k,v in row.items() if k not in ['ticks','corpus_seconds']}=={
            k:v for k,v in nodes['candidate'][name].items() if k not in ['ticks','corpus_seconds']},name
    graphs={label:matched_graphs.Graph(original,observed=value) for label,value in nodes.items()}
    cuts=[r for r in matched['rows'] if r['kind'] in ['attention-mask','attention-cleanup','convolution-mask']]
    assert len(cuts)==72 and len({r['name'] for r in cuts})==72
    assert Counter(r['kind'] for r in cuts)==Counter({'attention-mask':24,'attention-cleanup':24,'convolution-mask':24})
    families={};unions={label:{} for label in graphs};rows=[];membership={label:Counter() for label in graphs}
    for cut in cuts:
        name=cut['name'];kind=cut['kind'];family=families.setdefault(kind,{label:{} for label in graphs})
        assert name==f"/layers.{cut['layer']}/"+{'attention-mask':'self_attn/Where','attention-cleanup':'self_attn/Where_1','convolution-mask':'conv/Where'}[kind]
        for label,g in graphs.items():
            node=g.nodes[name]
            assert node['op']=='Where' and node['calls']==60 and node['outputs']==[cut['output']]
            assert not g.attrs(node) and len(node['inputs'])==3
            boundary={g.condition(node['inputs'][0]),node['inputs'][2]}
            assert boundary==set(cut['boundary'])
            assert g.constant(node['inputs'][1]).tobytes().hex()==cut['true_scalar_bits']
            found=g.ancestors(cut['output'],boundary)
            family[label].update(found);unions[label].update(found);membership[label].update(found.keys())
        rows.append(dict(kind=kind,layer=cut['layer'],name=name,boundary=cut['boundary'],output=cut['output'],
            calls=60,selected_seconds=nodes['selected'][name]['corpus_seconds'],candidate_seconds=nodes['candidate'][name]['corpus_seconds']))
    family_rows=[]
    for kind,family in families.items():
        assert family['selected'].keys()==family['candidate'].keys()
        expected=matched['families'][kind]['managed']
        assert len(family['selected'])==expected['nodes']
        assert Counter(n['op'] for n in family['selected'].values())==Counter({op:v['nodes'] for op,v in expected['operators'].items()})
        totals={label:matched_graphs.summarize(value) for label,value in family.items()}
        kernels={label:sum(n['corpus_seconds'] for n in value.values() if n['op']=='Where') for label,value in family.items()}
        assert all(v>0 for v in kernels.values())
        family_rows.append(dict(kind=kind,totals=totals,kernels=kernels,
            kernel_gain=1-kernels['candidate']/kernels['selected'],
            complete_group_gain=1-totals['candidate']['seconds']/totals['selected']['seconds']))
    assert unions['selected'].keys()==unions['candidate'].keys()
    union_rows=[dict(name=name,op=row['op'],members=membership['selected'][name],
        selected_seconds=row['corpus_seconds'],candidate_seconds=unions['candidate'][name]['corpus_seconds'])
        for name,row in unions['selected'].items()]
    assert all(membership['selected'][r['name']]==membership['candidate'][r['name']] for r in union_rows)
    assert sum(r['op']=='Where' for r in union_rows)==72
    totals={label:matched_graphs.summarize(value) for label,value in unions.items()}
    before=sum(r['selected_seconds'] for r in rows);after=sum(r['candidate_seconds'] for r in rows)
    return dict(kernels=72,layers=24,selected_seconds=before,candidate_seconds=after,gain=1-after/before,
        complete_groups=totals,complete_group_gain=1-totals['candidate']['seconds']/totals['selected']['seconds'],
        families=family_rows,rows=rows,union_rows=union_rows,shared_ancestors_counted_once=True)
