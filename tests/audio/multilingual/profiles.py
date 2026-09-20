"""Explicit prospective policies; host changes do not rewrite older evidence."""
def policy(name):
    if name=='windows':return dict(runtime='.NET 10.0.12',rss=20*1024**3,seconds=3600,available=1024**3,preflight=20*1024**3)
    if name=='amd':return dict(runtime='.NET 10.0.8',rss=14*1024**3,seconds=3600,available=1024**3,preflight=13*1024**3)
    raise ValueError('Unknown host profile: '+str(name))


def check_runtime(name,result):
    assert result['runtime']==policy(name)['runtime'] and result['affinity']==4 and not result['flags']
