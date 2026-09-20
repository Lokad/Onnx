"""Conditional label sensitivity on closed identical work; never a replacement timing verdict."""
from pathlib import Path
import argparse,json,math,shutil,time
import numpy as np
from audit import CASES,PERMUTATIONS,pin,read,write,evaluate

ROOT=Path(__file__).resolve().parents[3]
REPLICATES=1000

def orders(design,seed):
    state=seed
    def shuffle(values):
        nonlocal state
        values=list(values)
        for i in range(len(values)-1,0,-1):
            state=(state*1664525+1013904223)&0xffffffff;j=state%(i+1);values[i],values[j]=values[j],values[i]
        return values
    if design=='global24':return shuffle(list(range(6))*4)
    if design=='local6':return [x for _ in range(4) for x in shuffle(range(6))]
    if design=='mirror12':
        result=[]
        for _ in range(2):
            part=shuffle(range(6));result+=part+part[::-1]
        return result
    raise ValueError(design)

def analyze(batch,labels):
    # batch shape:case,visit,cycle,position,boundary; all calls enter each mean.
    mask=labels[...,None,None]==np.arange(3)[None,None,None,None,None,None,:]
    weighted=batch[None,...,None]*mask
    means=weighted.sum(axis=(2,3,4))/96
    workers=weighted.sum(axis=(3,4))/24
    positions=weighted.sum(axis=(2,3))/32
    aggregate=[];visit=[];position=[];contrast=[];raw=[]
    for numerator,denominator in [(1,0),(2,0),(2,1)]:
        ar=means[...,numerator]/means[...,denominator]
        vr=workers[...,numerator]/workers[...,denominator]
        pr=positions[...,numerator]/positions[...,denominator]
        cr=pr.max(axis=2)/pr.min(axis=2)
        aggregate.append((ar>=.995)&(ar<=1.005));visit.append(((vr>=.99)&(vr<=1.01)).all(axis=2))
        position.append(((pr>=.99)&(pr<=1.01)).all(axis=2));contrast.append(cr<=1.01)
        raw.append(dict(aggregate=ar.tolist(),workers=vr.tolist(),positions=pr.tolist(),contrast=cr.tolist()))
    gates={name:np.logical_and.reduce(values) for name,values in [('aggregate',aggregate),('workers',visit),('positions',position),('position_contrast',contrast)]}
    accepted=np.logical_and.reduce(list(gates.values()))
    return dict(all_cases_both_boundaries=accepted.all(axis=(1,2)).tolist(),per_case_both_boundaries=accepted.all(axis=2).tolist(),
                gates={k:v.tolist() for k,v in gates.items()},pairs=raw)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    assert not out.exists();out.mkdir()
    original=ROOT/'artifacts/e5-fingerprint-model-20260920';receipt=read(original/'closed.json')
    assert pin(original/'closed.json')['sha256']=='97e3ecd9230f3aa169da51805be565bc789569c3864f88cad0c2ecea727e3ce9'
    assert {p.relative_to(original).as_posix() for p in original.rglob('*') if p.is_file()}==set(receipt['files'])|{'closed.json'}
    for name,wanted in receipt['files'].items():assert pin(original/name)==wanted,name
    payload=original/'collected-aa';records=[];batch=np.zeros((5,4,24,3,2),dtype=np.float64);original_labels=np.zeros((1,5,4,24,3),dtype=np.int8)
    for index,case in enumerate(CASES):
        for visit in range(4):
            value=read(payload/f'result-aa/v{visit}-{case}/output/result.json');records.append(value)
            assert all(row['enabled'] is False for row in value['measured'])
            calls=value['schedule']['calls'];rows=value['measured'];assert len(rows)==24*3*calls
            for cycle in range(24):
                for position in range(3):
                    selected=rows[(cycle*3+position)*calls:(cycle*3+position+1)*calls]
                    assert len({r['role'] for r in selected})==1
                    original_labels[0,index,visit,cycle,position]=selected[0]['role']
                    for boundary,key in enumerate(['execute','request']):batch[index,visit,cycle,position,boundary]=math.fsum(r[key]/value['frequency'] for r in selected)/calls
    reconstructed=analyze(batch,original_labels);official=evaluate(records,'aa')
    assert official==read(original/'aa-audit.json')['timing'] and not official['passed']
    for index,item in enumerate(official['cases']):
        for boundary,key in enumerate(['execute','request']):
            for pair,wanted in enumerate(item['boundaries'][key]['controls']):
                observed=reconstructed['pairs'][pair]
                assert abs(observed['aggregate'][0][index][boundary]-wanted['ratio'])<1e-12
                assert abs(observed['contrast'][0][index][boundary]-wanted['position_contrast'])<1e-12
                assert np.allclose(np.array(observed['workers'][0][index])[:,boundary],wanted['workers'],atol=1e-12,rtol=0)
                assert np.allclose(np.array(observed['positions'][0][index])[:,boundary],wanted['positions'],atol=1e-12,rtol=0)
    summaries={};perms=np.array(PERMUTATIONS,dtype=np.int8)
    for design in ['global24','local6','mirror12']:
        labels=np.empty((REPLICATES,5,4,24,3),dtype=np.int8);balance=0
        for replicate in range(REPLICATES):
            for index in range(5):
                for visit in range(4):
                    order=orders(design,20260920+100*visit+index+10000*replicate)
                    assert sorted(order)==sorted(list(range(6))*4)
                    labels[replicate,index,visit]=perms[order]
                    for position in range(3):
                        assert np.bincount(labels[replicate,index,visit,:,position],minlength=3).tolist()==[8,8,8]
                    if design=='mirror12':
                        for start in [0,12]:
                            for position in range(3):
                                for role in range(3):
                                    indices=np.flatnonzero(labels[replicate,index,visit,start:start+12,position]==role)
                                    assert len(indices)==4 and indices.mean()==5.5
                                    # Exact arithmetic sum cancels a linear chronological drift within this block.
                                    assert sum(1000+3*int(i)+position for i in indices)==4*(1000+3*5.5+position)
                        balance+=1
        if design=='global24':assert np.array_equal(labels[:1],original_labels)
        value=analyze(batch,labels);write(out/(design+'.json'),value)
        summaries[design]=dict(replicates=REPLICATES,all_case_acceptances=sum(value['all_cases_both_boundaries']),
            per_case_acceptances=np.sum(value['per_case_both_boundaries'],axis=0).tolist(),
            gate_acceptances={k:int(np.array(v).all(axis=(1,2)).sum()) for k,v in value['gates'].items()},
            mirrored_balance_checks=balance,result=pin(out/(design+'.json')))
    np.save(out/'all-batch-means.npy',batch);write(out/'original-analysis.json',reconstructed)
    shutil.copyfile(Path(__file__),out/'source.py');shutil.copyfile(Path(__file__).with_name('audit.py'),out/'audit-source.py')
    write(out/'summary.json',dict(passed=True,created=time.time(),source_receipt=pin(original/'closed.json'),samples=16704,replicates_per_design=REPLICATES,
          designs=summaries,interpretation='Conditional relabeling sensitivity, not new execution, independent replicates, future-host probability, or a replacement verdict'))
    files={p.name:pin(p) for p in out.iterdir() if p.is_file()};write(out/'closed.json',dict(files=files,passed=True,source_receipt=pin(original/'closed.json')))
    print(json.dumps(summaries))

if __name__=='__main__':main()
