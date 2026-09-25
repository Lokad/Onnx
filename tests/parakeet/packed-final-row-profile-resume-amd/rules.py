"""Permit resumption only after the exact pre-launch memory refusal."""


def verify_refusal(state,spec):
    assert state['kind']=='capture' and state['complete'] and state['code']==1
    assert [r['name'] for r in state['runs']]==['control','phase']
    control,phase=state['runs']
    assert control['complete'] and control['code']==0 and control['members'] and control['samples']>0
    assert set(phase)=={'name','command','complete','code','members','samples','preflight'}
    assert not phase['complete'] and phase['code'] is None and phase['members']=={} and phase['samples']==0
    limits=spec['capture_limits']
    assert phase['preflight']['available']<limits['available_before']
    assert phase['preflight']['tmpfs']>=limits['tmpfs_before']
    assert "assert preflight['available']>=limits['available_before']" in state['error']
    return dict(passed=True,completed=['control'],unstarted=['phase','wall'],
        memory_shortfall=limits['available_before']-phase['preflight']['available'],
        original_owner=state['supervisor'])
