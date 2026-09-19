"""Independently compare every saved managed frontend value with native NPY data."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    result=json.loads(args.result.read_text(encoding='utf-8'))
    assert result['passed'] and result['comparisons']==26 and result['rejections']==1
    assert result['manifest_sha256']==sha(args.manifest)
    assert [s['name'] for s in result['steps']]==[s['name'] for s in manifest['cases']]
    rows=[]
    for step,reference in zip(result['steps'],manifest['cases']):
        if 'expected_failure' in reference:
            assert step['expected_failure'] and step['error']
            continue
        assert {o['name'] for o in step['outputs']}==set(reference['outputs'])
        for output in step['outputs']:
            file=reference['outputs'][output['name']]
            native=args.manifest.parent/file
            assert sha(native)==manifest['files'][file]['sha256']
            want=np.load(native,allow_pickle=False)
            raw=Path(str(args.result)+'.tensors')/output['raw_file']
            assert sha(raw)==output['sha256'] and raw.stat().st_size==want.nbytes
            got=np.fromfile(raw,dtype=want.dtype).reshape(want.shape)
            assert output['shape']==list(want.shape) and np.isfinite(got).all()
            if want.dtype==np.float32:
                error=float(np.max(np.abs(got.astype(np.float64)-want)/np.maximum(1,np.abs(want.astype(np.float64))),initial=0))
                assert error<=1e-4 and error==output['max_scaled_error'] and output['failed_values']==0
            else:
                assert want.dtype==np.int64 and np.array_equal(got,want)
                error=0
            rows.append(dict(case=step['name'],output=output['name'],values=want.size,max_scaled_error=error,sha256=sha(raw)))
    assert len(rows)==26
    report=dict(passed=True,result_sha256=sha(args.result),manifest_sha256=sha(args.manifest),outputs=len(rows),
        values=sum(r['values'] for r in rows),max_scaled_error=max(r['max_scaled_error'] for r in rows),rows=rows)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print('PASS',report['outputs'],'outputs;',report['values'],'values; max scaled error',report['max_scaled_error'])

if __name__=='__main__': main()
