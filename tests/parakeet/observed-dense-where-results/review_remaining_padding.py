"""Reconcile remaining Pad work from closed profiles; no model execution or candidate."""
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import onnx

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-observed-dense-where-remaining-padding-20260924'
PROFILE=ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924'
MATCHED=ROOT/'artifacts/parakeet-masking-padding-attribution-20260924'
WORK=ROOT/'artifacts/parakeet-masking-padding-work-counts-20260924'
BUILD=ROOT/'artifacts/parakeet-observed-dense-where-inventory-amd-20260924'
SOURCE=ROOT/'artifacts/parakeet-observed-dense-where-source-20260924'
MODEL=ROOT/'models/parakeet-tdt-0.6b-v3/encoder-model.onnx'
HELPER=ROOT/'tests/parakeet/observed-dense-where-profile-amd/groups.py'
spec=importlib.util.spec_from_file_location('observed_mask_groups',HELPER)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
Graph=module.matched_graphs.Graph
summarize=module.matched_graphs.summarize


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def closed(base,digest=None):
    if digest is not None:assert pin(base/'closed.json')['sha256']==digest
    proof=read(base/'closed.json');assert proof['passed'] and proof['analysis']==pin(base/'analysis.json')
    return read(base/'analysis.json')


def compare(phases,matched,original):
    nodes={role:{r['name']:r for r in p['node_rows'] if r['graph']=='encoder'} for role,p in phases.items()}
    assert set(nodes)=={'selected','candidate'}
    assert all(len(rows)==2856 for rows in nodes.values()) and nodes['selected'].keys()==nodes['candidate'].keys()
    for name,before in nodes['selected'].items():
        assert {k:v for k,v in before.items() if k not in ['ticks','corpus_seconds']}=={
            k:v for k,v in nodes['candidate'][name].items() if k not in ['ticks','corpus_seconds']},name
    graphs={role:Graph(original,observed=rows) for role,rows in nodes.items()}
    cuts=[r for r in matched['rows'] if r['kind'] in ['attention-pad','convolution-pad']]
    assert len(cuts)==48 and len({r['name'] for r in cuts})==48
    families={};unions={role:{} for role in nodes};rows=[]
    membership={role:Counter() for role in nodes}
    for cut in cuts:
        kind=cut['kind'];name=cut['name']
        suffix='self_attn/Pad' if kind=='attention-pad' else 'conv/depthwise_conv/Pad'
        assert name==f"/layers.{cut['layer']}/"+suffix
        family=families.setdefault(kind,{role:{} for role in nodes})
        for role,g in graphs.items():
            n=g.nodes[name]
            assert n['op']=='Pad' and n['calls']==60 and n['outputs']==[cut['output']]
            assert len(n['inputs'])==3 and not n['inputs'][2] and g.attrs(n)=={'mode':b'constant'}
            boundary={n['inputs'][0]};assert boundary==set(cut['boundary'])
            assert g.constant(n['inputs'][1]).tolist()==cut['pads']
            group=g.ancestors(cut['output'],boundary)
            family[role].update(group);unions[role].update(group);membership[role].update(group.keys())
        rows.append(dict(kind=kind,layer=cut['layer'],name=name,calls=60,
            selected_seconds=nodes['selected'][name]['corpus_seconds'],
            candidate_seconds=nodes['candidate'][name]['corpus_seconds'],
            earlier_ort_group_seconds=cut['ort_seconds']))
    totals=[]
    for kind,family in families.items():
        assert sum(r['kind']==kind for r in rows)==24
        assert family['selected'].keys()==family['candidate'].keys()
        expected=matched['families'][kind]['managed']
        assert len(family['selected'])==expected['nodes']
        assert Counter(r['op'] for r in family['selected'].values())==Counter({k:v['nodes'] for k,v in expected['operators'].items()})
        totals.append(dict(kind=kind,complete_groups={role:summarize(group) for role,group in family.items()},
            kernels={role:sum(r['corpus_seconds'] for r in group.values() if r['op']=='Pad') for role,group in family.items()},
            earlier_ort_group=matched['families'][kind]['ort']))
    assert unions['selected'].keys()==unions['candidate'].keys()
    assert membership['selected']==membership['candidate']
    union_rows=[dict(name=name,op=r['op'],members=membership['selected'][name],
        selected_seconds=r['corpus_seconds'],candidate_seconds=unions['candidate'][name]['corpus_seconds'])
        for name,r in unions['selected'].items()]
    assert sum(r['op']=='Pad' for r in union_rows)==48
    return dict(rows=rows,families=totals,union_rows=union_rows,
        complete_groups={role:summarize(group) for role,group in unions.items()},
        kernels={role:sum(r[role+'_seconds'] for r in rows) for role in nodes},shared_ancestors_counted_once=True)


def main():
    assert not BASE.exists()
    profile=closed(PROFILE,'8a6f509a210641650a9c057f472417dde7b4acc320f86eeb3d34ecbced4cc2e6')
    matched=closed(MATCHED,'0299d2e1d273b2651d45209cb7c9dffb623ae4149cf0cb5eed23eaf89d842e76')
    work=closed(WORK);build=closed(BUILD,'7d5296b266feeacbf9c52bf57a689de2cdf0a839045130f15912912f5da1b7bd')
    assert profile['requests']==160 and profile['clips']==20 and len(profile['frames'])==19
    assert profile['original_request_checks'] and profile['attribution_only'] and not profile['overhead_subtracted']
    assert build['inventory']['changed_method'].startswith('Lokad.Onnx.CPUExecutionProvider::Where::')
    assert build['inventory']['all_existing_flags_exact'] and build['inventory']['unchanged_core_methods']==3250
    assert build['source_prepared']==pin(SOURCE/'prepared.json')
    source=read(SOURCE/'prepared.json');shape='src/Lokad.Onnx/CPUExecutionProvider.Shape.cs'
    assert source['before'][shape]==source['source'][shape]==pin(ROOT/shape)==matched['managed_sources'][shape]
    assert pin(MODEL)==matched['original_model'] and work['attribution']==pin(MATCHED/'closed.json')
    revision=matched['ort_revision'];assert revision==work['ort_revision']=='2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
    native={}
    for path in ['onnxruntime/core/providers/cpu/tensor/pad.cc','onnxruntime/core/providers/cpu/tensor/utils.h']:
        data=subprocess.check_output(['git','-c','gc.auto=0','-C',str(ROOT/'external/onnxruntime'),'show',revision+':'+path])
        native[path]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
        assert native[path]==matched['native_sources'][path]==work['native_sources'][path]
    value=compare(profile['phases'],matched,onnx.load(MODEL,load_external_data=False).graph)
    value.update(passed=True,profile=pin(PROFILE/'closed.json'),matched=pin(MATCHED/'closed.json'),
        work=pin(WORK/'closed.json'),build=pin(BUILD/'closed.json'),source=pin(SOURCE/'prepared.json'),
        managed_pad_source=pin(ROOT/shape),native_sources=native,ort_revision=revision,
        source_counts=work['totals']['pad'],candidate_profile_seconds=profile['phases']['candidate']['corpus_seconds'],
        profile_pad_fraction=value['kernels']['candidate']/profile['phases']['candidate']['corpus_seconds'],
        helper=pin(HELPER),graph_helper=pin(Path(module.matched_graphs.__file__)),
        new_inference_calls=0,new_optimization_selected=False,overhead_subtracted=False,
        ort_profile_is_earlier=True,current_release_qualification_pending=True)
    documents=[OUT/f'remaining-padding-20260924{s}' for s in ['.json','.csv','.md']]
    assert not any(p.exists() for p in documents)
    BASE.mkdir()
    (BASE/'analysis.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf8')
    (BASE/'closed.json').write_text(json.dumps(dict(passed=True,analysis=pin(BASE/'analysis.json'),
        analyzer=pin(Path(__file__)),new_inference_calls=0),indent=2)+'\n',encoding='utf8')
    with documents[0].open('x',encoding='utf8') as f:json.dump(dict(closure=pin(BASE/'closed.json'),**value),f,indent=2,allow_nan=False)
    with documents[1].open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(value['rows'][0]));writer.writeheader();writer.writerows(value['rows'])
    lines=['# Remaining Parakeet padding cost after the masking change','',
        f"The retained candidate profile still spends **{value['kernels']['candidate']:.6f} seconds**",
        f"inside all 48 encoder Pad operations, or **{100*value['profile_pad_fraction']:.3f}%** of its",
        'profiled complete corpus time. This is attribution with observer overhead,',
        'not a new application score or an achievable speedup prediction.','',
        '| Complete padding family | Current profile (s) | Masking candidate profile (s) | Earlier ORT profile (s) |',
        '|---|---:|---:|---:|']
    for row in value['families']:
        lines.append(f"| {row['kind']} | {row['complete_groups']['selected']['seconds']:.6f} | {row['complete_groups']['candidate']['seconds']:.6f} | {row['earlier_ort_group']['seconds']:.6f} |")
    lines += ['',f"Complete groups contain {len(value['union_rows'])} unique nodes; shared input construction is counted once.",
        'The boundaries include pad-parameter construction and exclude the preceding',
        'data-producing matrix or convolution operation. Every layer, shape, call',
        'and original node matches the previously qualified graph correspondence.',
        'The ORT column is retained historical profiling of the same logical work;',
        'it is not a fresh comparison against this candidate. No clocks are pooled.','',
        'The exact installed ORT revision dispatches four-byte elements through',
        '`PadImpl<uint32_t>`, copies contiguous innermost rows and fills their borders.',
        'Lokad still fills the whole output, then calculates every source element’s',
        'destination through per-axis division and remainder. Its Pad source,',
        'compiled bodies and existing method flags are unchanged by the masking edit.','',
        'For the observed attention `[1,8,T,2T-1]` and convolution `[1,1024,T]`',
        'shapes, the retained source counts are 815,859,456 coordinate-loop steps',
        'over 220,412,352 input values. ORT’s source instead traverses 1,005,504',
        'contiguous copy blocks. These are source counts, not measured instructions',
        'or memory traffic; allocator clearing and compiler transformations are excluded.','',
        'The earlier row-copy prototype improved its eligible synthetic shapes but',
        'failed fallback and repeatability gates. Those failures remain rejected.',
        'This review supports padding as the next bounded diagnosis after the current',
        'release qualification. It does not admit that prototype, select a new variant',
        'or authorize bypassing any correctness or application performance gate.','',
        '[All 48 operations](remaining-padding-20260924.csv),',
        '[complete membership, source identities and counts](remaining-padding-20260924.json),',
        '[actual layouts](../managed-phase-results/masking-layouts-20260924.md),',
        '[earlier rejected screen](../pad-first-use-results/screen-20260923.md).','',
        'No model ran and no product file changed. Closure: `'+pin(BASE/'closed.json')['sha256']+'`.']
    with documents[2].open('x',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps(dict(passed=True,closed=pin(BASE/'closed.json'),kernels=value['kernels'],
        complete_groups=value['complete_groups'],profile_pad_fraction=value['profile_pad_fraction'])))


if __name__=='__main__':main()
