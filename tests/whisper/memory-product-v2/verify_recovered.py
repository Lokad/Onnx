"""Recheck closure, every displayed budget and actual test/resource totals."""
import xml.etree.ElementTree as ET,zipfile
from common import *


def main():
    base=ROOT/'artifacts/whisper-memory-product-v2-20260921';closed=read(base/'closed.json');assert closed['passed'] and closed['prototype_only']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    assert all(absent(b) for b in closed['births'])
    folder=Path(__file__).parent;value=read(folder/'observations-20260921.json');assert value==read(base/'audit.json')
    report=(folder/'results-20260921.md').read_text(encoding='utf-8');rows=0;normalizations=0
    for setting in SETTINGS:
        result=read(base/('consumer-corrected-'+setting['name']+'.json'))
        assert result['negativeBudgetRefusals']==1 and result['complete']
        for row in result['budgets']:
            assert row['cold']==64 and row['warm']==(64 if row['budget']<32 else 32) and row['ownership']
            assert row['first']==list(range(4,33,4)) and row['second']==list(range(80,109,4));rows+=1
            assert f"| {row['budget']} | {row['cold']} | {row['warm']} | Yes |" in report
        for case in result['cases']:
            x=np.asarray(case['input'],dtype=np.float32).astype(np.float64).reshape(3,case['width'])
            centered=x-x.mean(axis=1,keepdims=True);expected=centered/np.sqrt(np.mean(centered**2,axis=1,keepdims=True)+float(np.float32(1e-5)))*np.asarray(case['scales'],dtype=np.float32)
            if case['bias']:expected+=np.asarray(case['biases'],dtype=np.float32)
            actual=np.asarray(case['actual'],dtype=np.float32).reshape(x.shape)
            assert np.isfinite(actual).all() and np.max(np.abs(actual-expected)/np.maximum(1,np.abs(expected)))<=1e-6
            normalizations+=actual.size
    assert rows==16 and normalizations==40728 and '**40,728**' in report
    for name,expected in [('backend',3101),('tensors',342)]:
        root=ET.parse(base/'test-results'/(name+'.trx')).getroot();c=next(e for e in root.iter() if e.tag.endswith('Counters'))
        assert c.attrib['failed']=='0' and int(c.attrib['passed'])==expected and f'**{expected:,} ' in report
    samples=0;peak=0
    for prefix,state_name in [('', 'run.json'),('consumer-corrected-','consumer-corrected-run.json'),('il-stable-','il-stable-run.json')]:
        for run in read(base/state_name)['runs']:
            records=[json.loads(s) for s in (base/(prefix+run['name']+'.samples.jsonl')).read_text().splitlines()]
            assert len(records)==run['samples'];samples+=len(records)
            peak=max(peak,max(sum(m['rss'] for m in r['members']) for r in records))
    assert f'**{samples:,} resource samples**' in report and f'**{peak:,} bytes**' in report
    proof=read(base/'method-equivalence-stable.json');assert proof['passed'] and sum(v['methods'] for v in proof['observations'])==3777
    package=base/'source/artifacts/nuget/Lokad.Onnx.0.2.0.nupkg'
    assert pin(package)==value['package']==pin(base/'packages-corrected/lokad.onnx/0.2.0/lokad.onnx.0.2.0.nupkg')
    with zipfile.ZipFile(package) as archive:
        dll=archive.read('lib/net10.0/Lokad.Onnx.dll');assert hashlib.sha256(dll).hexdigest()==value['core']['sha256']
    write(base/'final-verification.json',dict(passed=True,closure=pin(base/'closed.json'),pins=len(closed['files']),budget_rows=rows,normalization_values=normalizations,resource_samples=samples,methods_compared=3777,births=closed['births']))
    print(json.dumps(dict(passed=True,closure=pin(base/'closed.json'),pins=len(closed['files']),budget_rows=rows,normalization_values=normalizations)))


if __name__=='__main__':main()
