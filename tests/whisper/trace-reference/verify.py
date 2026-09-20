"""Independent full-array recomputation of all retained chunked comparisons."""
import math,time
import numpy as np
from analyze import ROOT,BASE,pin,read,write,psutil


def main():
    import json
    started=time.monotonic();process=psutil.Process();inputs=read(BASE/'inputs.json')
    for name,wanted in inputs.items():assert pin(ROOT/name)==wanted,name
    rows=[json.loads(line) for line in (BASE/'comparisons.jsonl').read_text().splitlines()]
    assert len(rows)==1312
    maxima=[]
    for row in rows:
        a=np.fromfile(ROOT/row['actual'],dtype='<f4').astype(np.float64)
        b=np.fromfile(ROOT/row['expected'],dtype='<f8')
        assert a.size==b.size==math.prod(row['shape'])==row['values'] and np.isfinite(a).all() and np.isfinite(b).all()
        difference=a-b;absolute=np.abs(difference);relative=absolute/np.maximum(1,np.abs(b))
        assert float(relative.max())==row['max_scaled'] and int(relative.argmax())==row['scaled_flat_index']
        assert float(absolute.max())==row['max_absolute'] and int(absolute.argmax())==row['absolute_flat_index']
        assert int(np.sum(relative>0.0001))==row['failed_values']
        assert math.isclose(float(np.sqrt(np.mean(difference*difference))),row['rms'],rel_tol=1e-13,abs_tol=1e-18)
        resource=dict(seconds=time.monotonic()-started,rss=process.memory_info().rss,available=psutil.virtual_memory().available)
        assert resource['seconds']<900 and resource['rss']<2*1024**3 and resource['available']>=1024**3
        maxima.append(resource['rss'])
    observations=read(BASE/'observations.json')
    assert len(observations['paths'])==16
    for path in observations['paths']:
        for engine,v in path['references'].items():
            selected=[r for r in rows if (r['request'],r['kind'],r['reference'])==(path['request'],path['kind'],engine)]
            assert [r['index'] for r in selected]==list(range(41))
            bad=[r['index'] for r in selected if r['failed_values']]
            assert v['first_failure']==(bad[0] if bad else None) and v['failed_boundaries']==len(bad) and v['final']==selected[-1]
    write(BASE/'independent-verification.json',dict(passed=True,inputs=len(inputs),comparisons=len(rows),scalar_checks=7*len(rows),paths=16,seconds=time.monotonic()-started,peak_rss=max(maxima)))
    print(json.dumps(read(BASE/'independent-verification.json')))


if __name__=='__main__':main()
