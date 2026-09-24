"""Check every recorded Pad layout against the existing dispatcher and retain failures."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import onnx

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-observed-padding-routes-20260924'
LAYOUT=ROOT/'artifacts/parakeet-masking-padding-layout-amd-20260924'
PROFILE=ROOT/'artifacts/parakeet-observed-dense-where-profile-resume-amd-20260924'
SOURCE=ROOT/'artifacts/parakeet-pad-dispatch-source-20260923'
OLD=ROOT/'artifacts/parakeet-pad-dispatch-screen-amd-20260923'
FRONTEND=ROOT/'models/parakeet-tdt-0.6b-v3/nemo128.onnx'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def main():
    assert not BASE.exists()
    assert pin(LAYOUT/'closed.json')['sha256']=='3884b8ab0d36f8e6f2344bd5a3e7d711c19bebbf6c91bdc402d64d1fe8ac2ac8'
    lp=read(LAYOUT/'closed.json');assert lp['passed']
    assert pin(LAYOUT/'capture-collected/capture-collection.json')==lp['collection']
    collected=read(LAYOUT/'capture-collected/capture-collection.json')
    assert collected['terminal'] and collected['code']==0
    assert pin(PROFILE/'closed.json')['sha256']=='8a6f509a210641650a9c057f472417dde7b4acc320f86eeb3d34ecbced4cc2e6'
    proof=read(PROFILE/'closed.json');assert proof['passed'] and proof['analysis']==pin(PROFILE/'analysis.json')
    profile=read(PROFILE/'analysis.json')
    manifest=read(ROOT/'artifacts/parakeet-observed-dense-where-app-amd-20260924/collected/manifests/current-parakeet.json')
    assert pin(FRONTEND)=={k:manifest['models']['nemo128.onnx'][k] for k in ['bytes','sha256']}
    graph=onnx.load(FRONTEND,load_external_data=False).graph
    node,=[n for n in graph.node if n.op_type=='Pad']
    assert node.name=='n6_2' and list(node.input)==['waveforms_10','tmp_11'] and list(node.output)==['waveforms_12']
    assert {a.name:onnx.helper.get_attribute_value(a) for a in node.attribute}=={'mode':b'reflect'}
    pads,=[t for t in graph.initializer if t.name=='tmp_11']
    assert onnx.numpy_helper.to_array(pads).tolist()==[0,256,0,256]
    frontend={};encoder={}
    for role,phase in profile['phases'].items():
        rows=[r for r in phase['node_rows'] if r['op']=='Pad'];assert len(rows)==49
        other,=[r for r in rows if r['graph']!='encoder']
        assert (other['graph'],other['name'],other['calls'])==('frontend','n6_2',60)
        assert other['inputs']==list(node.input) and other['outputs']==list(node.output)
        assert all(r['calls']==60 for r in rows)
        frontend[role]=other['corpus_seconds'];encoder[role]=sum(r['corpus_seconds'] for r in rows if r['graph']=='encoder')
    counts=Counter();frames=set();observations=[]
    for index in range(80):
        relative=f'phase/layout-{index:03}.json';path=LAYOUT/'capture-collected'/relative
        assert pin(path)==collected['files'][relative]
        call,=read(path)['Calls'];rows=[r for r in call['Records'] if r['Op']=='Pad']
        assert len(rows)==48
        for row in rows:
            data=row['Inputs'][0];shape=data['Dimensions'];rank=len(shape);pads=row['Pads']
            assert row['Mode']=='constant' and row['FillSource']=='default' and row['FillBits']=='00000000'
            assert rank in [3,4] and len(pads)==2*rank
            assert data['ExactDense'] and not data['Reversed'] and data['ArrayBacked'] and data['ArrayOffset']==0
            assert data['StorageLength']==data['Length']
            # Exact existing dispatch guards, evaluated on retained observations.
            eligible=(row['Mode']!='reflect' and not data['Reversed'] and pads[rank-1]>=0 and pads[2*rank-1]>=0
                and all(pads[a]==pads[rank+a]==0 for a in range(rank-1)))
            assert eligible
            family=row['Family'];assert family in ['attention-pad','convolution-pad']
            t=shape[2] if family=='attention-pad' else shape[-1];frames.add(t)
            assert shape==([1,8,t,2*t-1] if family=='attention-pad' else [1,1024,t])
            assert pads==([0,0,0,1,0,0,0,0] if family=='attention-pad' else [0,0,4,0,0,4])
            counts[family]+=1
            observations.append(dict(request=index,name=row['Name'],family=family,frames=t,eligible=eligible))
    assert counts==Counter({'attention-pad':1920,'convolution-pad':1920}) and len(frames)==19
    prepared=read(SOURCE/'prepared.json');assert prepared['passed']
    helper='src/Lokad.Onnx/Zzz.LastAxisPadDispatch.cs'
    assert pin(SOURCE/'source'/helper)==prepared['source'][helper]==pin(ROOT/'tests/parakeet/pad-dispatch-source/Zzz.LastAxisPadDispatch.cs')
    old=read(OLD/'closed.json');assert old['passed'] and not old['admitted'] and old['files']['analysis.json']==pin(OLD/'analysis.json')
    result=dict(passed=True,layout=pin(LAYOUT/'closed.json'),profile=pin(PROFILE/'closed.json'),
        frontend_model=pin(FRONTEND),frontend_node=node.name,frontend_mode='reflect',frontend_pads=[0,256,0,256],
        frontend_profile_seconds=frontend,encoder_profile_seconds=encoder,
        frontend_fraction_of_candidate_profile=frontend['candidate']/profile['phases']['candidate']['corpus_seconds'],
        selected_layout_observations=len(observations),families=dict(counts),frames=sorted(frames),observations=observations,
        existing_helper=prepared['source'][helper],existing_source=pin(SOURCE/'prepared.json'),
        rejected_dispatch_screen=pin(OLD/'closed.json'),rejected_screen_verdict_unchanged=True,
        eligibility_is_source_prediction_on_prior_layout_capture=True,new_candidate_layout_capture=False,
        new_inference_calls=0,new_optimization_selected=False,application_gain_claimed=False)
    documents=[OUT/f'padding-routes-20260924{s}' for s in ['.json','.md']]
    assert not any(p.exists() for p in documents)
    BASE.mkdir();(BASE/'analysis.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    (BASE/'closed.json').write_text(json.dumps(dict(passed=True,analysis=pin(BASE/'analysis.json'),analyzer=pin(Path(__file__)),new_inference_calls=0),indent=2)+'\n',encoding='utf8')
    with documents[0].open('x',encoding='utf8') as f:json.dump(dict(closure=pin(BASE/'closed.json'),**result),f,indent=2)
    text=f'''# Parakeet padding routes: actual workload and retained screen failures

The complete profile has **49 Pad nodes**. All 48 encoder nodes use positive
constant padding on the final axis. The remaining frontend node is reflection
padding: `nemo128.onnx`, `n6_2`, widths `[0,256,0,256]`.

The masking-candidate profile attributes **{frontend['candidate']:.9f} s** to
that frontend node, versus **{encoder['candidate']:.9f} s** to the encoder Pad
nodes. The frontend share is {100*result['frontend_fraction_of_candidate_profile']:.4f}%
of the profiled complete corpus. Each node has all 60 measured calls. These
profile clocks include observer overhead and are not a new application score.

All **3,840 recorded encoder Pad layouts**, covering 80 requests, 24 layers and
19 frame counts, satisfy the existing row-copy dispatcher's source guards.
These are the earlier qualified layout observations. This review does not
claim a fresh layout capture on the masking candidate or execute the dispatcher.
The reflection frontend would still use the original fallback.

The old dispatcher screen remains **rejected**: cropping, outer padding and
reflection regressed, and six repeatability controls failed. The complete
Parakeet workload does not execute its synthetic cropping or outer-padding
cases, but it does execute reflection. Its smaller measured contribution does
not establish that a changed dispatcher preserves its latency or compilation
state. That boundary needs explicit scrutiny in any future complete-model test.

This narrows the next diagnosis to encoder row copying and the actual frontend
reflection fallback. It does not admit the old prototype, change a numerical
or performance limit, or select another compiler-flag variation. Current
masking release qualification remains the prerequisite for a new experiment.

[Every observation and source identity](padding-routes-20260924.json),
[remaining complete padding groups](remaining-padding-20260924.md),
[original rejected dispatcher screen](../pad-dispatch-results/screen-20260923.md),
[retained runtime diagnosis](../pad-runtime-diagnostic-results/report-20260923.md).

No inference ran and no product changed. Closure: `{pin(BASE/'closed.json')['sha256']}`.
'''
    with documents[1].open('x',encoding='utf8') as f:f.write(text)
    print(json.dumps({k:v for k,v in result.items() if k not in ['observations']}))


if __name__=='__main__':main()
