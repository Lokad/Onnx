"""Extend qualification to all new store widths before the first timing run."""
from pathlib import Path
import sys

TOOLS=Path(__file__).resolve().parent
ORIGINAL=TOOLS.parent / 'direct-output'
sys.path.insert(0,str(ORIGINAL))
import common
import prepare as original_preparation

common.BASE=common.ROOT / 'artifacts/pyannote-direct-output-store-v2-20260922'
common.REMOTE='/dev/shm/lokad-pyannote-direct-output-store-v2-20260922'
common.monitor.BASE=common.BASE
original_geometry=original_preparation.geometry


def geometry(census):
    result=original_geometry(census)
    assert len(result['cases'])==2882
    for pattern in ['finite','zero','special']:
        for m in [32,64]:
            for n in [64,65]:
                for k in [3,4,5,6,35,36,37,38]:
                    for stride,start in [(k,0),(k+11,3)]:
                        for bias in [False,True]:
                            result['cases'].append(dict(m=m,n=n,k=k,stride=stride,start=start,bias=bias,pattern=pattern))
    assert len(result['cases'])==3266
    return result


original_preparation.geometry=geometry


def main():
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect','audit']
    mode=sys.argv[1]
    prior=common.ROOT / 'artifacts/pyannote-direct-output-v4-20260922'
    assert common.pin(prior / 'closed.json')['sha256']=='7bfa418ab3ba005217c34dcf2ac80f859ca77c822e2fb662e779eff0912ce318'
    closed=common.read(prior / 'closed.json'); assert closed['passed']; common.verify(closed['files'])
    analysis=common.read(prior / 'analysis.json')
    assert not analysis['eligible'] and analysis['controls_passed'] and analysis['geomean_candidate_baseline']<=.95
    assert [(r['m'],r['n'],r['k']) for r in analysis['rows'] if r['candidate_baseline']>1.05]==[(256,1152,2),(256,2304,2)]
    path=ORIGINAL / 'complete_v4.py'; source=path.read_text(encoding='utf8')
    assert source.count('pyannote-direct-output-v4-20260922')==2
    source=source.replace('pyannote-direct-output-v4-20260922','pyannote-direct-output-store-v2-20260922')
    assert source.count('from generate_v4 import generate')==3
    source=source.replace('from generate_v4 import generate','from store_generator import generate')
    old="folder in [c.TOOLS,c.BASE / 'consumer',normal_folder,payload,original,c.ROOT / 'artifacts/pyannote-direct-output-v2-20260922',c.ROOT / 'artifacts/pyannote-direct-output-v3-20260922']"
    assert source.count(old)==1
    source=source.replace(old,old[:-1]+",STORE_TOOLS,PRIOR_STORE,c.ROOT / 'artifacts/pyannote-direct-output-store-20260922']")
    changes=[("    assert c.read(original / 'payload/shapes.json')==expected", "    prior_shapes=c.read(original / 'payload/shapes.json')\n    assert prior_shapes['shapes']==expected['shapes'] and prior_shapes['cases']==expected['cases'][:2882]"),
        ("    shutil.copy2(original / 'payload/shapes.json',payload / 'shapes.json')", "    c.save(payload / 'shapes.json',expected)"),
        ("assert result['passed'] and len(result['records'])==2882", "assert result['passed'] and len(result['records'])==3266"),
        ('cases_per_mode=2882','cases_per_mode=3266')]
    for old,new in changes:
        assert source.count(old)==1,old; source=source.replace(old,new)
    namespace=dict(__name__='complete_ordinary_store_successor',__file__=str(path),STORE_TOOLS=TOOLS,PRIOR_STORE=prior)
    exec(compile(source,str(path),'exec'),namespace)
    if mode in ['prepare','audit']:
        namespace[mode]()
        if mode=='prepare':
            current=common.read(common.BASE / 'payload/shapes.json'); old=common.read(prior / 'payload/shapes.json')
            assert current['shapes']==old['shapes'] and current['cases'][:2882]==old['cases']
            assert common.pin(common.BASE / 'normal/Probe.cs')==common.pin(prior / 'normal/Probe.cs')
    else:
        import transport
        getattr(transport,mode)()


if __name__=='__main__': main()
