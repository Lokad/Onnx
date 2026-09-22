"""Use the closed v4 protocol with a distinct two-column routine."""
from pathlib import Path
import sys

TOOLS=Path(__file__).resolve().parent
ORIGINAL=TOOLS.parent / 'direct-output'
sys.path.insert(0,str(TOOLS.parent / 'direct-output-store'))
sys.path.insert(0,str(ORIGINAL))
import common
import prepare as original_preparation
import complete as store_complete

# Reuse the complete 3266-case grid, not the ordinary-store generator.
assert original_preparation.geometry is store_complete.geometry
common.BASE=common.ROOT / 'artifacts/pyannote-two-column-20260922'
common.REMOTE='/dev/shm/lokad-pyannote-two-column-20260922'
common.monitor.BASE=common.BASE


def main():
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect','audit']
    mode=sys.argv[1]
    diagnostic=common.ROOT / 'artifacts/pyannote-tail-codegen-20260922'
    assert common.pin(diagnostic / 'closed.json')['sha256']=='a47e0e2a234d0bed861dde60a2453af81376773513fe76fbc154c27851594e78'
    common.verify(common.read(diagnostic / 'closed.json')['files'])
    path=ORIGINAL / 'complete_v4.py'; source=path.read_text(encoding='utf8')
    changes=[('pyannote-direct-output-v4-20260922','pyannote-two-column-20260922',2),
        ('from generate_v4 import generate','from two_column_generator import generate',3),
        ("normal.count('if (Avx2.IsSupported)')==1", "normal.count('if (Avx2.IsSupported)')==2",1),
        ("    assert c.read(original / 'payload/shapes.json')==expected", "    prior_shapes=c.read(original / 'payload/shapes.json')\n    assert prior_shapes['shapes']==expected['shapes'] and prior_shapes['cases']==expected['cases'][:2882]",1),
        ("    shutil.copy2(original / 'payload/shapes.json',payload / 'shapes.json')", "    c.save(payload / 'shapes.json',expected)",1),
        ("assert result['passed'] and len(result['records'])==2882", "assert result['passed'] and len(result['records'])==3266",1),
        ('cases_per_mode=2882','cases_per_mode=3266',1)]
    for old,new,count in changes:
        assert source.count(old)==count,(old,source.count(old)); source=source.replace(old,new)
    old="folder in [c.TOOLS,c.BASE / 'consumer',normal_folder,payload,original,c.ROOT / 'artifacts/pyannote-direct-output-v2-20260922',c.ROOT / 'artifacts/pyannote-direct-output-v3-20260922']"
    assert source.count(old)==1
    source=source.replace(old,old[:-1]+',NEW_TOOLS]')
    namespace=dict(__name__='two_column_successor',__file__=str(path),NEW_TOOLS=TOOLS)
    exec(compile(source,str(path),'exec'),namespace)
    if mode in ['prepare','audit']:
        namespace[mode]()
        if mode=='prepare':
            prior=common.ROOT / 'artifacts/pyannote-direct-output-store-v2-20260922'
            assert common.read(common.BASE / 'payload/shapes.json')==common.read(prior / 'payload/shapes.json')
            assert common.pin(common.BASE / 'normal/Probe.cs')==common.pin(prior / 'normal/Probe.cs')
    else:
        import transport
        getattr(transport,mode)()


if __name__=='__main__':main()
