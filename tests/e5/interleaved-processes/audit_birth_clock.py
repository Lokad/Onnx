"""Correct only the frozen auditor's comparison of unlike Linux time bases.

/proc/stat btime is integer seconds; psutil create_time adds clock ticks to it.
Compare worker and supervisor births within that same domain. Recorded wall-clock
launch/finish ordering remains checked separately. No producer or gate changes.
The original frozen audit.py remains unchanged and is shipped with the evidence.
"""
import audit as original
from audit import *
frozen_telemetry=original.telemetry

def telemetry(state,records,jobs,smoke=False):
    assert state['complete'] is True and state['code']==0 and state['limits']==LIMITS
    assert [r['job'] for r in state['runs']]==jobs
    previous=state['started'];births=[state['supervisor']];summary=[]
    for run in state['runs']:
        job=run['job'];assert run['complete'] is True and run['code']==0
        assert previous<=run['started']<run['ended']<=state['ended'];previous=run['ended']
        assert 0<run['seconds']<LIMITS['seconds'] and list(run['workers'])==job['creation']
        assert [(e['role'],e['op'],e['index']) for e in run['events']]==commands(job,smoke)
        last=0.
        for event in run['events']:
            assert last<=event['started']<event['ended']<=run['seconds'];last=event['ended']
            assert event['ack']==[event['op'],event['index']]
        for role,worker in run['workers'].items():
            assert worker['code']==0 and worker['birth']>=state['supervisor']['birth']
            assert run['started']<=worker['started']<worker['ended']<=run['ended']
            births.append(dict(pid=worker['pid'],birth=worker['birth']))
        seen=set();paused={};last=-1.;peak=0;minimum=2**64
        samples=records[job['name']];assert len(samples)==run['samples']>0
        for row in samples:
            assert last<=row['seconds']<=run['seconds'];last=row['seconds']
            assert row['available']>=LIMITS['available'];minimum=min(minimum,row['available'])
            assert len({m['role'] for m in row['members']})==len(row['members'])
            assert row['active'] is None or row['active'] in ROLES
            for member in row['members']:
                role=member['role'];worker=run['workers'][role];seen.add(role)
                assert member['pid']==worker['pid'] and member['birth']==worker['birth'] and member['affinity']==[2]
                assert member['rss']>=0 and math.isfinite(member['cpu_seconds']) and member['cpu_seconds']>=0
                if role!=row['active']:
                    assert member['suspended'] is True
                    if not smoke:assert member['status']=='stopped'
                    if role not in paused:paused[role]=member['cpu_seconds']
                    assert -.000001<=member['cpu_seconds']-paused[role]<=.02
                else:paused.pop(role,None)
            rss=sum(m['rss'] for m in row['members']);assert rss<LIMITS['rss'];peak=max(peak,rss)
        assert seen==set(ROLES) and peak==run['peak_rss']
        summary.append(dict(job=job['name'],seconds=run['seconds'],samples=len(samples),peak_rss=peak,minimum_available=minimum))
    assert len({(b['pid'],b['birth']) for b in births})==len(births)
    return dict(births=births,cohorts=summary)

original.telemetry=telemetry

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--payload',type=Path,required=True);p.add_argument('--phase',choices=['aa','compare'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();result=original.audit(a.payload.resolve(),a.phase);write(a.output,result)
    print(json.dumps(dict(passed=True,timing_passed=result['timing']['passed'],measured_calls=result['measured_calls'],solo_calls=result['solo_calls'])))
