"""Narrow the next diagnostic using exact source and retained complete-call evidence."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
REV='2e2543fbe9fae542f921d47a72d21d5a4ef0b710'


def read(path):return json.loads(path.read_text(encoding='utf8'))


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def main():
    routes_path=ROOT/'tests/parakeet/observed-dense-where-results/projection-routes-20260924.json'
    routes=read(routes_path)
    calls_path=ROOT/'artifacts/parakeet-projection-route-resume-amd-20260924/matched-projection-calls.json'
    assert routes['passed'] and routes['no_new_inference'] and not routes['actual_kernel_leaf_observed']
    assert pin(calls_path)==routes['every_matched_call']
    for name,wanted in routes['source_files'].items():assert pin(ROOT/'src/Lokad.Onnx'/name)==wanted,name
    calls=[r for r in read(calls_path) if '/feed_forward' in r['name']]
    assert len(calls)==96*80 and len({r['name'] for r in calls})==96
    assert all(not r['positional'] and r['copy_bytes']==0 and r['m']==r['frames'] for r in calls)
    buckets=defaultdict(list)
    for row in calls:
        assert (row['k'],row['n'],row['alpha']) in [(1024,4096,1.),(4096,1024,.5)]
        assert row['scratch_bytes']==(0 if row['route']=='mapped-allowed' else 16777216)
        if row['pass_index']>0:buckets[(row['k'],row['n'],row['alpha'],row['route'])].append(row)
    groups=[]
    for key,values in sorted(buckets.items()):
        groups.append(dict(k=key[0],n=key[1],alpha=key[2],route=key[3],
            weights=len({r['name'] for r in values}),calls_per_corpus=len(values)//3,
            scratch_requested_per_corpus=sum(r['scratch_bytes'] for r in values)//3,
            frames=sorted({r['frames'] for r in values}),copy_bytes=0))
    assert sum(r['calls_per_corpus'] for r in groups)==1920
    assert sum(r['scratch_requested_per_corpus'] for r in groups)==30249320448
    assert [(r['k'],r['n'],r['route'],r['weights']) for r in groups]==[
        (1024,4096,'mapped-allowed',9),(1024,4096,'mapped-declined',9),
        (1024,4096,'unmapped',39),(4096,1024,'unmapped',48)]
    gap=read(OUT/'remaining-gap-20260925.json');assert gap['passed'] and gap['no_new_inference']
    candidate_groups=[r for r in gap['projection_groups'] if r['nodes']==48]
    assert len(candidate_groups)==2
    native_report=ROOT/'tests/parakeet/ort-diagnosis-results/ort-kernel-observations-20260924.json'
    native=read(native_report);assert native['passed'] and native['source_revision']==REV
    sources={}
    for name in ['onnxruntime/core/providers/cpu/math/matmul.cc','onnxruntime/core/providers/cpu/math/gemm.cc']:
        blob=subprocess.check_output(['git','-C',str(ROOT/'external/onnxruntime'),'show',REV+':'+name])
        text=blob.decode('utf8')
        needles=['Status MatMul<float>::PrePack','Status MatMul<float>::Compute','data[i].BIsPacked = bool(packed_b_);'] if name.endswith('matmul.cc') else ['bool GemmPackBFp32','MlasGemmPackBSize','MlasGemmPackB(']
        assert all(needle in text for needle in needles)
        sources[name]=dict(bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest(),git_object=subprocess.check_output(['git','-C',str(ROOT/'external/onnxruntime'),'rev-parse',REV+':'+name],text=True).strip(),lines={needle:text[:text.index(needle)].count('\n')+1 for needle in needles})
    rejected=ROOT/'artifacts/parakeet-inclusive-packing-app-amd-20260924'
    proof=read(rejected/'closed.json');analysis=read(rejected/'analysis.json')
    assert pin(rejected/'closed.json')['sha256']=='5a7344ce05b9d2d8c6cf8cd8c7a618030a594424fe160a645a088bdf96a600dc'
    assert proof['passed'] and not proof['admitted'] and not analysis['performance']['admitted']
    assert proof['files']['analysis.json']==pin(rejected/'analysis.json')
    failed=[r for r in analysis['performance']['gates'] if not r['passed']]
    assert len(failed)==1 and all(r['passed'] for r in analysis['performance']['controls'])
    value=dict(passed=True,no_new_inference=True,no_product_candidate_selected=True,
        current_candidate_profile=pin(OUT/'remaining-gap-20260925.json'),candidate_groups=candidate_groups,
        earlier_route_capture=pin(routes_path),earlier_matched_calls=pin(calls_path),route_groups=groups,
        route_capture_precedes_slice_candidate=True,source_dispatch_unchanged=routes['source_files'],
        native_revision=REV,native_kernel_proof=pin(native_report),native_sources=sources,
        native_prepack_buffers_not_dumped=True,packing_time_not_yet_separated=True,
        prior_inclusive_trial=dict(closure=pin(rejected/'closed.json'),analysis=pin(rejected/'analysis.json'),admitted=False,failed_gates=failed),
        reviewer=pin(Path(__file__)))
    with (OUT/'feed-forward-review-20260925.json').open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2,allow_nan=False)
    print(json.dumps(dict(passed=True,groups=groups,candidate_profile_excess=sum(r['excess'] for r in candidate_groups),new_candidate_selected=False)))


if __name__=='__main__':main()
