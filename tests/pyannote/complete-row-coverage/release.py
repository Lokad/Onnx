from common import *

def released():
    finish=ROOT/'artifacts/pyannote-convolution-portable-comparison-finish-20260921'
    prepared=read(finish/'prepared.json');st=read(finish/'state.json')
    assert st['supervisor']==dict(pid=969536,birth=1790021268.89557)
    assert st['complete'] and st['code']==0 and st['phase']=='complete', 'Current local comparison still owns the lane or failed.'
    terminal(st['supervisor']);terminal(prepared['application'])
    assert [r['name'] for r in st['stages']]==prepared['stages']
    for row in st['stages']:
        assert row['complete'] and row['code']==0
        for pid,birth in row['members'].items():terminal(dict(pid=int(pid),birth=birth))
    for key,folder in [('application_closure','pyannote-convolution-portable-applications-20260921'),
                       ('comparison_closure','pyannote-convolution-portable-comparison-20260921')]:
        p=ROOT/'artifacts'/folder/'closed.json'
        assert pin(p)==st[key] and read(p)['passed']
    return finish/'state.json'
