"""Recompute the prospective screen from every retained AMD sample."""
from pathlib import Path
import argparse,json,math,statistics
from remote import h,processes

ROWS=[8,30,128,512]
PROJECTIONS=[(384,384),(384,1536),(1536,384)]

def samples(row):
    values=row['samples_ms']
    assert len(values)==7 and all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in values)
    return statistics.median(values)

def performance(runs):
    assert len(runs)==4 and [r['order'] for r in runs]==[1,2,3,4]
    medians=[]
    for run in runs:
        rows=run['bank']
        assert len(rows)==12 and {(r['m'],r['mode']) for r in rows}=={(m,mode) for m in ROWS for mode in range(3)}
        medians.append({(r['m'],r['mode']):samples(r) for r in rows})
    summaries=[]
    controls=True;gains=True;regressions=True
    for m in ROWS:
        duplicate=[r[m,1]/r[m,0] for r in medians]
        duplicate_mean=statistics.geometric_mean(duplicate)
        control_ok=.98<=duplicate_mean<=1.02 and all(.95<=v<=1.05 for v in duplicate)
        controls &= control_ok
        comparisons=[]
        for control in (0,1):
            ratios=[r[m,2]/r[m,control] for r in medians]
            mean=statistics.geometric_mean(ratios)
            if m in (30,128):
                gain_ok=mean<=.98 and all(v<=1.02 for v in ratios)
                gains &= gain_ok
            else:
                gain_ok=mean<=1.01;regressions &= gain_ok
            comparisons.append(dict(control=control,ratios=ratios,geometric_mean=mean,passes=bool(gain_ok)))
        summaries.append(dict(m=m,duplicate_ratios=duplicate,duplicate_geometric_mean=duplicate_mean,control_passes=bool(control_ok),
            median_ms=[{str(mode):r[m,mode] for mode in range(3)} for r in medians],comparisons=comparisons))
    return dict(controls_pass=bool(controls),primary_gain_pass=bool(gains),other_rows_pass=bool(regressions),
        nominated=bool(controls and gains and regressions),bank=summaries)

def run_audit(run,core,producer):
    assert run['checked_cases']==256 and run['packing_checks']==54 and run['refusals']==8
    assert run['runtime']=='10.0.8' and run['processor_count']==1 and run['affinity']==4 and run['avx512'] is True and run['flags']=={}
    assert run['core_sha256']==core and run['producer_sha256']==producer
    expected={(m,n,96,False) for m in list(range(8,46))+[64,128,512] for n in [1,65,129,257,385,513]}
    expected|={(m,385,192,True) for m in [8,13,14,15,27,28,29,30,42,128]}
    assert len(run['checks'])==256 and {(r['m'],r['n'],r['k'],r['exceptional']) for r in run['checks']}==expected
    assert len(run['results'])==36
    assert {(r['m'],r['n'],r['k'],r['mode']) for r in run['results']}=={(m,n,k,mode) for m in ROWS for n,k in PROJECTIONS for mode in range(3)}
    for row in run['results']+run['bank']:
        samples(row)
        assert len(row['gc_before'])==len(row['gc_after'])==3 and all(0<=a<=b for a,b in zip(row['gc_before'],row['gc_after']))
        assert row['allocated_bytes']>=0
    for m in ROWS:
        bank=[r for r in run['bank'] if r['m']==m]
        assert len(bank)==3 and all(r['matrices']==72 and r['packed_bytes']==84934656 and len(r['output_sha256'])==72 for r in bank)
        assert all(r['output_sha256']==bank[0]['output_sha256'] for r in bank)
        for n,k in PROJECTIONS:
            same=[r for r in run['results'] if (r['m'],r['n'],r['k'])==(m,n,k)]
            assert len(same)==3
            for key in ['a_sha256','b_sha256','packed_sha256','output_sha256']:
                assert len({r[key] for r in same})==1
            assert all(r['iterations']==(64 if m<64 else 16 if m<256 else 4) for r in same)

def resource_audit(identity,collected,collection):
    assert identity['complete'] and 'error' not in identity and identity['supervisor']['affinity']=='0'
    assert identity['limits']==dict(rss=2*1024**3,seconds=1200,available_memory=256*1024**2)
    assert [r['name'] for r in identity['runs']]==['1','2','3','4','codegen']
    terminals={(identity['supervisor']['pid'],identity['supervisor']['start'])}
    resources=[];end=identity['started']
    for row in identity['runs']:
        assert row['code']==0 and end<=row['started']<row['ended']<=identity['ended'] and 0<row['seconds']<1200
        end=row['ended'];last=-1;peak=0;available=math.inf;births={}
        entries=[json.loads(line) for line in (collected/('result/'+row['name']+'-samples.jsonl')).read_text().splitlines()]
        assert len(entries)==row['samples'] and entries
        for sample in entries:
            assert last<sample['seconds']<row['seconds'];last=sample['seconds']
            h.check_sample(sample,row['pid'],row['start'])
            peak=max(peak,sum(m['rss'] for m in sample['members']));available=min(available,sample['available_memory'])
            for member in sample['members']:
                assert births.get(str(member['pid']),member['start'])==member['start']
                births[str(member['pid'])]=member['start']
        assert births==row['members'] and births[str(row['pid'])]==row['start'] and peak==row['peak_rss']
        terminals.update((int(pid),start) for pid,start in births.items())
        account=processes.foreign_fraction(h.read(collected/('result/'+row['name']+'-pre.json')),h.read(collected/('result/'+row['name']+'-post.json')),identity['supervisor']['pid'])
        assert account==row['accounting']
        if row['name']!='codegen':assert row['flags']=={}
        resources.append(dict(name=row['name'],seconds=row['seconds'],samples=len(entries),peak_rss=peak,minimum_available=available,accounting=account))
    assert terminals=={(p['pid'],p['start']) for p in collection['terminal_processes']}
    assert collection['complete'] and collection['code']==0 and collection['checkout']=='172181fc5ab4eb2bdc2eb7f37e80d25e482a0887'
    return resources

def main():
    p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();assert not a.output.exists();base=a.artifact.resolve();payload=base/'payload';collected=base/'collected'
    h.verify(payload)
    bundle=h.read(payload/'bundle.json');collection=h.read(collected/'collection.json');download=h.read(base/'download.json')
    assert h.sha(collected/'collection.json')==download['collection_sha256']
    assert {p.relative_to(collected).as_posix() for p in collected.rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,pin in collection['files'].items():h.verify_file(h.safe_path(collected,name),pin)
    for name,pin in bundle['files'].items():
        if name in collection['files']:h.verify_file(collected/name,pin)
    assert h.sha(collected/'bundle.json')==h.sha(payload/'bundle.json')
    identity=h.read(collected/'result/identity.json');deployment=h.read(collected/'deployment.json')
    assert identity['bundle_sha256']==deployment['bundle_sha256']==h.sha(payload/'bundle.json')
    assert (deployment['pid'],deployment['start'])==(identity['supervisor']['pid'],identity['supervisor']['start'])
    assert int((collected/'complete.txt').read_text())==0
    resources=resource_audit(identity,collected,collection)
    runs=[h.read(collected/('result/'+str(i)+'.json')) for i in range(1,5)]
    for run in runs:run_audit(run,h.sha(payload/'bin/Lokad.Onnx.dll'),h.sha(payload/'bin/Probe.dll'))
    assert all(r['checks']==runs[0]['checks'] for r in runs)
    result=dict(schema=1,correctness_pass=True,performance=performance(runs),measured_batches=1344,geometry_checks=1024,packing_checks=216,refusals=32,
        core_sha256=h.sha(payload/'bin/Lokad.Onnx.dll'),producer_sha256=h.sha(payload/'bin/Probe.dll'),
        bundle_sha256=h.sha(payload/'bundle.json'),collection_sha256=h.sha(collected/'collection.json'),archive_sha256=download['sha256'],
        resources=resources,runs=runs,codegen_sha256=h.sha(collected/'result/jit.txt'))
    h.write_new(a.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['runs','resources']},indent=2))

if __name__=='__main__':main()
