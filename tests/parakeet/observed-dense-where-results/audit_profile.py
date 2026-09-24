"""Reconcile historical decoder packing metadata without changing graph checks."""
import copy
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
TOOLS=ROOT/'tests/parakeet/observed-dense-where-profile-resume-amd'
sys.path.insert(0,str(TOOLS))
from run import BASE,ORIGINAL,PRIOR,pin,read,write,prepared


def compare_graphs(current,candidate,historical):
    assert current==candidate
    assert historical['decoder_joint-model.onnx']['retained_packed_bytes']==25246720
    assert current['decoder_joint-model.onnx']['retained_packed_bytes']==51461120
    expected=copy.deepcopy(historical)
    expected['decoder_joint-model.onnx']['retained_packed_bytes']=51461120
    assert current==expected


def correction():
    original=TOOLS/'audit.py'
    assert pin(original)==read(BASE/'prepared.json')['tools']['audit.py']
    calls=ROOT/'artifacts/parakeet-prepared-recurrence-calls-amd-v2-20260924'
    assert pin(calls/'closed.json')['sha256']=='20b3bc4b5a2d2ddb6e0834e40fb7e2bcb5220dd1f45c813a7a830466ddbc463d'
    proof=read(calls/'closed.json');assert proof['passed']
    assert proof['analysis']==pin(calls/'analysis.json')
    reviews=read(calls/'analysis.json')['reviews']
    assert len(reviews)==4
    for review in reviews:
        assert review['passed']
        residency=next(r for r in review['residencies'] if r['name']=='decoder')
        expected=25246720 if review['role']=='selected' else 51461120
        assert residency['bytes']==sum(r['bytes'] for r in residency['weights'])==expected
    paths=[ORIGINAL/'capture-collected/control/graphs.json',BASE/'capture-collected/wall/graphs.json',PRIOR/'capture-collected/wall/graphs.json']
    compare_graphs(*(read(p) for p in paths))
    before="    assert read(first_folder/'control/graphs.json')==read(folder/'wall/graphs.json')==read(PRIOR/'capture-collected/wall/graphs.json')"
    after="    compare_graphs(read(first_folder/'control/graphs.json'),read(folder/'wall/graphs.json'),read(PRIOR/'capture-collected/wall/graphs.json'))"
    code=original.read_text(encoding='utf8');assert code.count(before)==1;code=code.replace(before,after)
    needle="write(BASE/'closed.json',dict(passed=passed,"
    assert code.count(needle)==1
    code=code.replace(needle,"write(BASE/'closed.json',dict(passed=passed,audit_correction=pin(BASE/'audit-correction.json'),")
    return code,dict(passed=True,original_auditor=pin(original),corrected_auditor=pin(Path(__file__)),
        recurrence_closure=pin(calls/'closed.json'),recurrence_analysis=pin(calls/'analysis.json'),
        graphs={p.relative_to(ROOT).as_posix():pin(p) for p in paths},
        field='decoder_joint-model.onnx.retained_packed_bytes',historical=25246720,current=51461120,
        added_recurrent_bytes=26214400,current_and_candidate_graph_metadata_exact=True,
        all_other_historical_graph_fields_exact=True,new_inference_calls=0,
        scope='Only historical decoder packing total is updated to the already qualified recurrence preparation. Every node, input, output and other graph field remains exact; all timing and correctness gates remain.')


def main():
    prepared();assert not (BASE/'closed.json').exists() and not (BASE/'audit-correction.json').exists()
    code,value=correction();write(BASE/'audit-correction.json',value)
    namespace=dict(__name__='corrected_profile_audit',__file__=str(Path(__file__).resolve()),compare_graphs=compare_graphs)
    exec(compile(code,str(TOOLS/'audit.py')+' [historical packing corrected]','exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
