"""Read-only intersection of the unchanged native and double-reference gates."""
import argparse, decimal
from shared import *


def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',required=True);a=p.parse_args();base=Path(a.artifact).resolve()
    spec=read(base/'manifest.json');audit=read(base/'audit.json');assert audit['diagnostic_passed'] and audit['manifest']==pin(base/'manifest.json')
    own=psutil_module().Process();own.cpu_affinity([0]);output=base/'compatibility';output.mkdir(exist_ok=False)
    records=[];total=0
    for case in spec['cases']:
        native=load_baseline(case['baselines']['native']).astype(np.float64);references=[]
        for name in ['numpy','torch']:
            path=base/name/case['name']/'features.npy';assert pin(path)==audit['files'][path.relative_to(base).as_posix()]
            references.append(np.load(path,allow_pickle=False))
        targets=[native,*references]
        lowers=[t-ORIGINAL_LIMIT*np.maximum(1,np.abs(t)) for t in targets]
        uppers=[t+ORIGINAL_LIMIT*np.maximum(1,np.abs(t)) for t in targets]
        gap=np.maximum.reduce(lowers)-np.minimum.reduce(uppers)
        indices=np.argwhere(gap>0);exact=[]
        with decimal.localcontext() as context:
            context.prec=90;D=decimal.Decimal;eps=D('0.0001')
            for index in indices:
                coord=tuple(map(int,index));numbers=[D.from_float(float(t[coord])) for t in targets]
                widths=[eps*max(D(1),abs(n)) for n in numbers]
                lower=max(n-w for n,w in zip(numbers,widths));upper=min(n+w for n,w in zip(numbers,widths))
                assert lower>upper and abs(float(lower-upper)-float(gap[coord]))<1e-13
                exact.append(dict(coordinate=list(coord),native=float(native[coord]),numpy=float(references[0][coord]),torch=float(references[1][coord]),
                                  exact_lower=str(lower),exact_upper=str(upper),exact_gap=str(lower-upper)))
        # Every omitted coordinate is at least 1e-12 away from a positive separation.
        assert np.max(gap[gap<=0]) < -1e-12
        path=output/(case['name']+'.npy')
        with path.open('xb') as f:np.save(f,gap,allow_pickle=False)
        records.append(dict(name=case['name'],values=int(gap.size),incompatible=len(exact),max_gap=float(gap.max()),
                            gaps=pin(path),coordinates=exact));total+=len(exact)
    result=dict(complete=True,diagnostic='post-hoc gate feasibility; no tolerance or acceptance change',audit=pin(base/'audit.json'),
                tolerance=ORIGINAL_LIMIT,total_values=sum(r['values'] for r in records),incompatible=total,records=records,
                source=pin(Path(__file__)),process=dict(pid=own.pid,birth=own.create_time()))
    write(base/'compatibility.json',result);print(json.dumps(dict(values=result['total_values'],incompatible=total,max_gap=max(r['max_gap'] for r in records))))


if __name__=='__main__':main()
