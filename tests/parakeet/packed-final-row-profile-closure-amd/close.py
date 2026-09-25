"""Collect completed profiles after the diagnosed status-path collision; never rerun."""
from collections import defaultdict
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/packed-final-row-profile-resume-amd'))
from checks import ORIGINAL, PROFILE, partial, original_checker, module
import resume

BASE = ROOT/'artifacts/parakeet-packed-final-row-profile-closure-amd-20260925'
pin, read, write = resume.pin, resume.read, resume.write


def verify_collision(state, checkpoint):
    assert state['kind'] == 'resume' and state['complete'] and state['code'] == 1
    assert [r['name'] for r in state['runs']] == ['phase', 'wall']
    assert all(r['complete'] and r['code'] == 0 and r['samples'] > 0 for r in state['runs'])
    expected_error = (
        'Traceback (most recent call last):\n'
        '  File "/dev/shm/lokad-parakeet-packed-final-row-profile-20260925/profile_resume.py", line 77, in main\n'
        "    verify();state['code']=0\n"
        '    ^^^^^^^^\n'
        '  File "/dev/shm/lokad-parakeet-packed-final-row-profile-20260925/profile_resume.py", line 32, in verify\n'
        "    assert original.pin(base/'capture-state.json')==details['original_state']\n"
        '           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n'
        'AssertionError\n')
    assert state['error'] == expected_error
    expected = dict(state, complete=False, code=None)
    del expected['ended']; del expected['error']
    assert checkpoint == expected, 'Only the final successful-run checkpoint is accepted'


def collect():
    resume.prepared(); old = partial()
    assert not BASE.exists()
    BASE.mkdir()
    write(BASE/'inputs.json', dict(driver=pin(__file__), resume_prepared=pin(resume.BASE/'prepared.json'),
        original_state=pin(PROFILE/'capture-collected/capture-state.json'),
        original_transfer=pin(PROFILE/'capture-transfer.json')))
    source = (ORIGINAL/'remote.py').read_text(encoding='utf8')
    assert source.count("save(BASE/'capture-state.json',state)") == 1
    original_ids = [old['state']['supervisor']] + [dict(pid=int(p), birth=b)
        for r in old['state']['runs'] for p, b in r['members'].items()]
    script = resume.PRELUDE + '\n' + inspect.getsource(verify_collision) + f'''
import io
from remote import read,pin,verify,live,idle
verify();idle()
details=read(base/'resume-spec.json')
assert pin(base/'resume-spec.json')=={pin(resume.BASE/'bundle/resume-spec.json')!r}
for name,wanted in details['files'].items():assert pin(base/name)==wanted,name
for name,key in [('spec.json','original_spec'),('remote.py','original_worker'),('capture-collection.json','original_collection'),('build-review.json','build_review'),('built.json','built')]:
 assert pin(base/name)==details[key],name
changed={{name:dict(expected=wanted,actual=pin(base/name)) for name,wanted in details['retained_files'].items() if pin(base/name)!=wanted}}
assert set(changed)=={{'capture-state.json'}}
state=read(base/'resume-state.json');checkpoint=read(base/'capture-state.json')
verify_collision(state,checkpoint)
assert state['supervisor']=={read(resume.BASE/'deployment.json')!r}
ids={original_ids!r}+[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert not any(live(i) for i in ids)
paths=sorted(p for p in base.rglob('*') if p.is_file() and ('logs' in p.parts or p.parent==base or p.parent.name in ['control','phase','wall']) and p.name!='transfer.tar.gz')
files={{p.relative_to(base).as_posix():pin(p) for p in paths}}
receipt=dict(files=files,state=pin(base/'resume-state.json'),terminal=True,code=state['code'],identities=ids,changed_retained=changed,
 supervisor_success=False,completed_processes_success=True,checkpoint_collision_verified=True)
raw=json.dumps(receipt,indent=2).encode()
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in files:archive.add(base/name,arcname=name,recursive=False)
 info=tarfile.TarInfo('closure-collection.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
'''
    with (BASE/'results.tar.gz').open('xb') as out, (BASE/'collection.stderr').open('x') as err:
        result = subprocess.run(resume.SSH+['python3','-B','-'], input=script.encode(), stdout=out,
            stderr=err, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode == 0, 'Preserve partial collection; no new inference'
    folder = BASE/'collected'; folder.mkdir()
    with tarfile.open(BASE/'results.tar.gz') as archive:
        members = archive.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len({m.name for m in members}) == len(members)
        archive.extractall(folder, filter='data')
    receipt = read(folder/'closure-collection.json')
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    write(BASE/'transfer.json', dict(passed=True, archive=pin(BASE/'results.tar.gz'), collection=pin(folder/'closure-collection.json')))
    print(json.dumps(dict(files=len(receipt['files']), supervisor_code=receipt['code'], completed_processes_success=True)))


def audit():
    resume.prepared(); old = partial()
    assert not (BASE/'closed.json').exists()
    inputs = read(BASE/'inputs.json')
    assert inputs['driver'] == pin(__file__)
    assert inputs['resume_prepared'] == pin(resume.BASE/'prepared.json')
    assert inputs['original_state'] == pin(PROFILE/'capture-collected/capture-state.json')
    assert inputs['original_transfer'] == pin(PROFILE/'capture-transfer.json')
    folder = BASE/'collected'; transfer = read(BASE/'transfer.json'); receipt = read(folder/'closure-collection.json')
    assert transfer['passed'] and transfer['archive'] == pin(BASE/'results.tar.gz')
    assert transfer['collection'] == pin(folder/'closure-collection.json')
    assert receipt['terminal'] and receipt['code'] == 1 and not receipt['supervisor_success']
    for name, wanted in receipt['files'].items(): assert pin(folder/name) == wanted, name
    changed = {name:dict(expected=wanted,actual=pin(folder/name)) for name,wanted in old['receipt']['files'].items() if pin(folder/name)!=wanted}
    assert set(changed) == {'capture-state.json'} and changed == receipt['changed_retained']
    assert pin(folder/'capture-collection.json') == pin(PROFILE/'capture-collected/capture-collection.json')
    previous = old['state']; state = read(folder/'resume-state.json')
    verify_collision(state, read(folder/'capture-state.json'))
    assert state['supervisor'] == read(resume.BASE/'deployment.json')
    assert state['original_state'] == inputs['original_state']
    assert 0 < state['ended']-state['started'] < 4*3600
    ids = [previous['supervisor']] + [dict(pid=int(p),birth=b) for r in previous['runs'] for p,b in r['members'].items()]
    ids += [state['supervisor']] + [dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    assert receipt['identities'] == ids and receipt['state'] == pin(folder/'resume-state.json')
    assert pin(folder/'resume-spec.json') == pin(resume.BASE/'bundle/resume-spec.json')
    details = read(folder/'resume-spec.json')
    for name, wanted in details['files'].items(): assert pin(folder/name) == wanted, name
    assert [r['mode'] for r in state['headroom_waits']] == ['phase','wall']
    spec = old['spec']; limits = spec['capture_limits']
    for wait in state['headroom_waits']:
        rows = wait['observations']; assert rows and all(0<=r['seconds']<900 for r in rows)
        assert all(0<=b['seconds']-a['seconds']<10 for a,b in zip(rows,rows[1:]))
        assert rows[-1]['available'] >= limits['available_before']+64*1024**2
        assert rows[-1]['tmpfs'] >= limits['tmpfs_before']
    assert pin(folder/'spec.json') == pin(PROFILE/'bundle/spec.json')
    assert pin(folder/'build-review.json') == pin(PROFILE/'build-review.json')
    assert read(folder/'built.json') == old['built']
    results={}; phases={}; resources=[]; check_run,_=original_checker()
    for owner, run in [(previous, previous['runs'][0]), *[(state,r) for r in state['runs']]]:
        check_run(run,owner,folder,spec,old['built'],old['expected'],old['protocol'],old['accounting'],old['manifest'],results,phases,resources)
    assert sum(len(v['records']) for v in results.values()) == 240
    assert sum(r['phase']=='measured' for v in results.values() for r in v['records']) == 180
    corpus = {mode:sum(r['seconds'] for r in v['records'] if r['phase']=='measured')/3 for mode,v in results.items()}
    assert read(folder/'phase/graphs.json') == read(folder/'wall/graphs.json')
    for mode, phase in phases.items(): assert math.isclose(phase['corpus_seconds'],corpus[mode],rel_tol=1e-14)
    operators=defaultdict(float)
    for row in phases['wall']['node_rows']: operators[(row['graph'],row['op'])]+=row['corpus_seconds']
    context = dict(spec['diagnostic_context'], original_capture_refused=True, completed_control_reused=True,
        resume_supervisor_success=False, resume_status_collision=True, completed_process_checks_passed=True)
    analysis = dict(passed=True, corpus=corpus, phase_over_control=corpus['phase']/corpus['control'],
        wall_over_phase=corpus['wall']/corpus['phase'], phases=phases, resources=resources,
        operators=[dict(graph=g,op=op,corpus_seconds=s) for (g,op),s in sorted(operators.items(),key=lambda p:-p[1])],
        original_request_checks=True,core_unchanged=True,consumer_unchanged=True,constructor_unchanged=True,
        complete_admitted_public_results_exact=True,source_receipt=spec['source_receipt'],
        qualification_closures=spec['qualification_closures'],diagnostic_context=context,
        original_refusal=old['refusal'],original_state=inputs['original_state'],resume_state=pin(folder/'resume-state.json'),
        overwritten_checkpoint=pin(folder/'capture-state.json'),resume_failure=state['error'],changed_retained=changed,
        headroom_waits=state['headroom_waits'],attribution_only=True,actual_kernel_dispatch_measured=False)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),build_review=pin(PROFILE/'build-review.json'),
        build_review_correction=pin(PROFILE/'build-review-correction.json'),original_transfer=inputs['original_transfer'],
        transfer=pin(BASE/'transfer.json'),collection=pin(folder/'closure-collection.json'),auditor=pin(__file__),terminal_owners=ids,
        supervisor_success=False,completed_process_checks_passed=True,
        files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(corpus=corpus,phase_over_control=analysis['phase_over_control'],wall_over_phase=analysis['wall_over_phase'],
        phases={k:{n:v[n] for n in ['phase_seconds','remainder_seconds','call_counts']} for k,v in phases.items()},resources=resources)))


def compare():
    proof=read(BASE/'closed.json');assert proof['passed']
    for name,wanted in proof['files'].items():assert pin(BASE/name)==wanted,name
    assert not (BASE/'comparison.json').exists()
    comparison=module('original_projection_comparison',ORIGINAL/'compare.py')
    write(BASE/'comparison-driver.json',dict(original=pin(ORIGINAL/'compare.py'),closure=pin(BASE/'closed.json'),
        only_override='BASE: completed processes after separately audited checkpoint collision'))
    comparison.BASE=BASE
    comparison.main()


if __name__ == '__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['collect','audit','compare']
    globals()[sys.argv[1]]()
