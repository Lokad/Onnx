"""Independently check full clustering arrays, partitions and canonical public assignments."""
from pathlib import Path
import argparse,hashlib,json,math
import numpy as np

def audit(reference,result):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    lfsha=lambda p:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    here=Path(__file__).resolve().parent
    r=json.loads(result.read_text(encoding='utf-8'));m=json.loads((reference/'reference.json').read_text(encoding='utf-8'))
    pins=json.loads((here/'pins.json').read_text(encoding='utf-8'))
    assert r['reference_sha256']==sha(reference/'reference.json')
    assert r['prepared_sha256']==m['prepared_sha256']==sha(reference/'prepared.json')
    assert m['generator_lf_sha256']==lfsha(here/'generate_reference.py') and m['prepare_lf_sha256']==lfsha(here.parents[2]/'eng/prepare-community1-plda.py')
    for k in ['source_revision','source_hashes','numpy','scipy']:assert m[k]==pins[k]
    for k in ['cases','hierarchy']:assert [c['name'] for c in m[k]]==pins[k]
    def array(e):
        assert e['dtype'] in ('float32','float64','int8','int32','int64')
        a=np.array(e['values'],np.float64);assert np.isfinite(a).all() and all(type(d) is int and d>=0 for d in e['shape'])
        if e['dtype'].startswith('int'):assert np.equal(a,np.trunc(a)).all()
        assert math.prod(e['shape'])==a.size
        return a.reshape(e['shape'])
    expected={};partitions={};public={}
    for c in m['hierarchy']:
        labels=array(c['labels']).reshape(-1);points=array(c['points']);z=array(c['hierarchy'])
        assert points.ndim==2 and z.shape==(points.shape[0]-1,4) and labels.shape==(points.shape[0],)
        expected[c['name'],'merge-distances']=z[:,2];partitions[c['name']]=labels
    for c in m['cases']:
        embeddings=array(c['embeddings']);activity=array(c['activity']);assert embeddings.ndim==3 and activity.ndim==3
        chunks,speakers,width=embeddings.shape;assert width==256 and activity.shape[0]==chunks and activity.shape[2]==speakers
        assert np.isin(activity,[0,1]).all();frames=activity.shape[1];assert frames>0
        clean=(activity*(activity.sum(axis=2,keepdims=True)==1)).sum(axis=1)
        training=np.flatnonzero((clean*5>=frames).reshape(-1));assert np.array_equal(training,array(c['training_indices']).reshape(-1))
        centers=array(c['centroids']);hard=array(c['hard']);soft=array(c['soft']);clusters=len(centers)
        assert centers.shape==(clusters,256) and hard.shape==(chunks,speakers) and soft.shape==(chunks,speakers,clusters)
        stages={'centroids':centers,'soft':soft,'hard':hard}
        if len(training)>1:
            labels=array(c['labels']);assert labels.shape==(len(training),)
            k=len(array(c['priors']));assert sorted(set(labels))==list(range(k));partitions[c['name']]=labels
            shapes={'normalized':(len(training),256),'transformed':(len(training),128),'responsibilities':(len(training),k),'priors':(k,),
                    'alpha':(k,128),'inverse_precision':(k,128)}
            for key,shape in shapes.items():
                a=array(c[key]);assert a.shape==shape;stages[key.replace('_','-')]=a
            objective=array(c['objective']);assert objective.ndim==2 and objective.shape[1]==1 and 2<=len(objective)<=20;stages['objective']=objective
        canonical={};labels=hard.astype(int).reshape(-1).copy();labels[~activity.any(axis=1).reshape(-1)]=-2
        for i,label in enumerate(labels):
            if label>=0:
                if label not in canonical:canonical[label]=len(canonical)
                labels[i]=canonical[label]
        for label in range(clusters):
            if label not in canonical:canonical[label]=len(canonical)
        ordered=np.empty_like(centers)
        for old,new in canonical.items():ordered[new]=centers[old]
        for repeat in range(2):
            stages['public-centroids-'+str(repeat)]=ordered
            public[c['name'],repeat]=dict(labels=labels.tolist(),clusters=clusters,training=len(training))
        for key,a in stages.items():expected[c['name'],key]=a.reshape(-1)
    for p in r['partitions']:
        e=partitions.pop(p['name']);labels=np.asarray(p['labels']);mapping=np.asarray(p['mapping'])
        assert labels.shape==e.shape and len(set(mapping.tolist()))==len(mapping) and sorted(set(labels.tolist()))==list(range(len(mapping)))
        assert np.array_equal(mapping[labels],e)
    assert not partitions
    for p in r['publicResults']:
        e=public.pop((p['name'],p['repeat']))
        for key in ('labels','clusters','training'):assert p[key]==e[key]
    assert not public
    total=bad_total=0;maximum=0.;rows=[];actuals={}
    for row in r['reports']:
        key=(row['name'],row['stage']);e=expected.pop(key);a=np.asarray(row['actual'],np.float64)
        assert a.shape==e.shape and np.isfinite(a).all()
        errors=np.abs(a-e)/np.maximum(1,np.abs(e));error=float(errors.max(initial=0));bad=int(np.count_nonzero(errors>pins['tolerance']))
        assert math.isclose(error,row['error'],rel_tol=1e-15,abs_tol=0) and bad==row['bad']
        if row['stage']=='hard':assert np.array_equal(a,e)
        if row['stage']=='public-centroids-1':assert np.array_equal(a,actuals[(row['name'],'public-centroids-0')])
        actuals[key]=a;total+=a.size;bad_total+=bad;maximum=max(maximum,error)
        rows.append(dict(name=row['name'],stage=row['stage'],maximum=error,bad=bad,values=a.size))
    assert not expected and total==r['values'] and r['passed']==(bad_total==0) and math.isclose(maximum,r['maximum'],rel_tol=1e-15,abs_tol=0)
    return dict(passed=bad_total==0,arrays=len(rows),values=total,maximum=maximum,bad=bad_total,
        pipeline_cases=len(m['cases']),hierarchy_cases=len(m['hierarchy']),public_requests=len(r['publicResults']),result_sha256=sha(result),rows=rows)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--result',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();answer=audit(args.reference,args.result)
    with args.output.open('x',encoding='utf-8') as f:json.dump(answer,f,indent=2)
    print(json.dumps({k:v for k,v in answer.items() if k!='rows'},indent=2));raise SystemExit(0 if answer['passed'] else 1)
