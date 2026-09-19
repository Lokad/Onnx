"""Independently check every saved managed filterbank value; numeric failures return1."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference',type=Path,required=True)
parser.add_argument('--result',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text(encoding='utf-8'))
source=args.reference/'manifest.json';manifest=read(source);result=read(args.result)
assert result['manifest_sha256']==sha(source)
pins=read(Path(__file__).with_name('pins.json'));assert manifest['pins']==pins and manifest['tolerance']==1e-4
assert [c['name'] for c in manifest['cases']]==pins['cases']
assert [r['name'] for r in result['reports']]==pins['cases']
for name,entry in manifest['files'].items():
    path=args.reference/name;assert path.parent.resolve()==args.reference.resolve() and sha(path)==entry['sha256']
    a=np.load(path,allow_pickle=False);assert a.dtype==np.float32 and list(a.shape)==entry['shape'] and np.isfinite(a).all()
rows=[];maximum=0;count=0;bad_total=0
for case,row in zip(manifest['cases'],result['reports']):
    expected=np.load(args.reference/case['output'],allow_pickle=False)
    directory=Path(str(args.result)+'.arrays');path=directory/row['file']
    assert path.parent.resolve()==directory.resolve() and sha(path)==row['sha256']
    actual=np.fromfile(path,np.float32).reshape(row['shape']);assert actual.shape==expected.shape and np.isfinite(actual).all()
    errors=np.abs(actual.astype(np.float64)-expected)/np.maximum(1,np.abs(expected.astype(np.float64)))
    error=float(errors.max());bad=int(np.count_nonzero(errors>1e-4))
    assert math.isclose(error,row['error'],rel_tol=1e-15,abs_tol=0) and bad==row['bad']
    assert int(errors.argmax())==row['worst_index']
    rows.append(dict(name=case['name'],values=actual.size,maximum=error,bad=bad));maximum=max(maximum,error);bad_total+=bad;count+=actual.size
passed=bad_total==0
assert result['passed']==passed and count==result['values'] and math.isclose(maximum,result['maximum'],rel_tol=1e-15,abs_tol=0)
answer=dict(passed=passed,arrays=len(rows),values=count,maximum=maximum,bad=bad_total,rows=rows,result_sha256=sha(args.result),manifest_sha256=sha(source))
with args.output.open('x',encoding='utf-8') as f:json.dump(answer,f,indent=2)
print(json.dumps({k:v for k,v in answer.items() if k!='rows'},indent=2))
raise SystemExit(0 if passed else 1)
