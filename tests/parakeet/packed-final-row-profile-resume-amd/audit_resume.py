"""Audit the retained control and two new profiles under their actual owners."""
from collections import defaultdict
import json
import math
from checks import partial,original_checker
from resume import BASE,PROFILE,prepared,pin,read,write


def main():
    prepared();assert not (BASE/'closed.json').exists()
    old=partial();folder=BASE/'collected'
    transfer=read(BASE/'transfer.json');receipt=read(folder/'resume-collection.json')
    assert transfer['passed'] and transfer['collection']==pin(folder/'resume-collection.json')
    assert transfer['archive']==pin(BASE/'results.tar.gz') and receipt['terminal'] and receipt['code']==0
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    # Every file collected after the refusal, including control clocks and the
    # failed state, must survive byte for byte in the new collection.
    for name,wanted in old['receipt']['files'].items():assert pin(folder/name)==wanted,name
    assert pin(folder/'capture-collection.json')==pin(PROFILE/'capture-collected/capture-collection.json')
    previous=read(folder/'capture-state.json');state=read(folder/'resume-state.json')
    assert previous==old['state']
    assert state['complete'] and state['code']==0 and state['kind']=='resume'
    assert state['supervisor']==read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']]==['phase','wall']
    assert state['original_state']==pin(folder/'capture-state.json')
    assert 0<state['ended']-state['started']<4*3600
    ids=[s['supervisor'] for s in [previous,state]]+[dict(pid=int(p),birth=b) for s in [previous,state] for r in s['runs'] for p,b in r['members'].items()]
    assert receipt['identities']==ids and receipt['state']==pin(folder/'resume-state.json')
    assert pin(folder/'resume-spec.json')==pin(BASE/'bundle/resume-spec.json')
    assert [r['mode'] for r in state['headroom_waits']]==['phase','wall']
    spec=old['spec'];limits=spec['capture_limits']
    for wait in state['headroom_waits']:
        rows=wait['observations'];assert rows and all(0<=r['seconds']<900 for r in rows)
        assert all(0<=b['seconds']-a['seconds']<10 for a,b in zip(rows,rows[1:]))
        assert rows[-1]['available']>=limits['available_before']+64*1024**2
        assert rows[-1]['tmpfs']>=limits['tmpfs_before']
    assert pin(folder/'spec.json')==pin(PROFILE/'bundle/spec.json')
    assert pin(folder/'build-review.json')==pin(PROFILE/'build-review.json')
    assert read(folder/'built.json')==old['built']
    results={};phases={};resources=[];check_run,_=original_checker()
    for owner,run in [(previous,previous['runs'][0]),*[(state,r) for r in state['runs']]]:
        check_run(run,owner,folder,spec,old['built'],old['expected'],old['protocol'],old['accounting'],old['manifest'],results,phases,resources)
    assert sum(len(v['records']) for v in results.values())==240
    assert sum(r['phase']=='measured' for v in results.values() for r in v['records'])==180
    corpus={mode:sum(r['seconds'] for r in value['records'] if r['phase']=='measured')/3 for mode,value in results.items()}
    assert read(folder/'phase/graphs.json')==read(folder/'wall/graphs.json')
    for mode,phase in phases.items():assert math.isclose(phase['corpus_seconds'],corpus[mode],rel_tol=1e-14)
    operators=defaultdict(float)
    for row in phases['wall']['node_rows']:operators[(row['graph'],row['op'])]+=row['corpus_seconds']
    context=dict(spec['diagnostic_context'],original_capture_refused=True,completed_control_reused=True)
    analysis=dict(passed=True,corpus=corpus,phase_over_control=corpus['phase']/corpus['control'],
        wall_over_phase=corpus['wall']/corpus['phase'],phases=phases,resources=resources,
        operators=[dict(graph=g,op=op,corpus_seconds=seconds) for (g,op),seconds in sorted(operators.items(),key=lambda p:-p[1])],
        original_request_checks=True,core_unchanged=True,consumer_unchanged=True,constructor_unchanged=True,
        complete_admitted_public_results_exact=True,source_receipt=spec['source_receipt'],
        qualification_closures=spec['qualification_closures'],diagnostic_context=context,
        original_refusal=old['refusal'],original_transfer=pin(PROFILE/'capture-transfer.json'),
        original_state=pin(folder/'capture-state.json'),resume_state=pin(folder/'resume-state.json'),
        headroom_waits=state['headroom_waits'],attribution_only=True,actual_kernel_dispatch_measured=False)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),build_review=pin(PROFILE/'build-review.json'),
        build_review_correction=pin(PROFILE/'build-review-correction.json'),
        transfer=pin(BASE/'transfer.json'),collection=pin(folder/'resume-collection.json'),
        original_transfer=pin(PROFILE/'capture-transfer.json'),auditor=pin(__file__),terminal_owners=ids,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(corpus=corpus,phase_over_control=analysis['phase_over_control'],wall_over_phase=analysis['wall_over_phase'],
        phases={k:{n:v[n] for n in ['phase_seconds','remainder_seconds','call_counts']} for k,v in phases.items()},operators=analysis['operators'][:16])))


if __name__=='__main__':main()
