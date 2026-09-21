"""Describe every retained resident-process control without revising its verdict."""
from pathlib import Path
import hashlib,importlib.util,itertools,json,math,shutil,statistics,sys

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'artifacts/e5-interleaved-processes-v3-20260920'
BASE=ROOT/'artifacts/e5-resident-variation-20260921'
RECEIPT='45fcf0832ef4f48b7f14c29fc5ea08b384b113122606dc18560e56bd5de73a17'
CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok']

def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())

def read(p):return json.loads(p.read_text(encoding='utf-8'))

def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def components(table):
    assert len(table)==3 and all(len(r)==48 for r in table)
    assert all(type(x) is int and x>=0 for r in table for x in r)
    rows=[sum(r) for r in table];cols=[sum(table[r][c] for r in range(3)) for c in range(48)];total=sum(rows)
    full=sum((144*x-total)**2 for row in table for x in row)
    process=48*sum((3*r-total)**2 for r in rows)
    cycle=3*sum((48*c-total)**2 for c in cols)
    residual=sum((144*table[r][c]-3*rows[r]-48*cols[c]+total)**2 for r in range(3) for c in range(48))
    assert full==process+cycle+residual
    return dict(total=full,process=process,cycle=cycle,residual=residual,
        shares={k:v/full if full else 0. for k,v in [('process',process),('cycle',cycle),('residual',residual)]})

def main():
    assert pin(SOURCE/'aa-closed.json')['sha256']==RECEIPT
    closed=read(SOURCE/'aa-closed.json');assert closed['evidence_passed'] and not closed['timing_passed']
    assert read(SOURCE/'aa-final-verification.json')['passed']
    assert not BASE.exists();BASE.mkdir()
    shutil.copyfile(ROOT/'.agent/m1-e5-resident-variation-20260921.md',BASE/'prospective-plan.md')
    inputs={}
    def source(name):
        path=SOURCE/name;wanted=closed['files'][name];assert pin(path)==wanted,name;inputs[str(path.relative_to(ROOT))]=wanted;return read(path)
    state=source('collected-aa/result-aa/identity.json');assert state['complete'] and len(state['runs'])==40
    prior=source('aa-audit.json');original={(r['case'],r['policy']):r for r in prior['timing']['cases']}
    workers=[];cohorts=[];counts=dict(measured=0,managed_measured=0,native_measured=0,conditioning=0,solo=0)
    for run in state['runs']:
        job=run['job'];members={}
        for role in ['A','B','C','N']:
            name='collected-aa/result-aa/'+job['name']+'/'+role+'/output/result.json';v=source(name);s=v['specification'];m=v['measured']
            assert v['passed'] and not v['enabled'] and not v['wide_enabled'] and s['phase']=='aa' and s['role']==role
            assert s['case_index']==job['case_index'] and s['policy']==job['policy'] and s['visit']==job['visit']
            assert s['blocks']==48 and s['calls']==[32,16,4,4,2][job['case_index']]
            assert [(r['block'],r['call']) for r in m]==list(itertools.product(range(48),range(s['calls'])))
            assert all(type(r[k]) is int and r[k]>0 for r in m for k in ['execute','request']) and all(r['request']>=r['execute'] for r in m)
            counts['measured']+=len(m);counts['native_measured' if role=='N' else 'managed_measured']+=len(m)
            counts['conditioning']+=len(v['conditioning']);counts['solo']+=len(v['solo']);members[role]=v
            workers.append(dict(cohort=job['name'],role=role,source=name,measured_calls=len(m),conditioning_calls=len(v['conditioning']),solo_calls=len(v['solo']),
                gc=[sum(r[k] for r in m) for k in ['g0','g1','g2']],calls_with_gc=sum(any(r[k] for k in ['g0','g1','g2']) for r in m),
                mean_allocated=statistics.fmean(r['bytes'] for r in m)))
        assert len({v['frequency'] for v in members.values()})==1
        for boundary in ['execute','request']:
            table=[[sum(r[boundary] for r in members[role]['measured'] if r['block']==b) for b in range(48)] for role in ['A','B','C']]
            decomposition=components(table);pairs=[]
            for numerator,denominator in [('B','A'),('C','A'),('C','B')]:
                n,d=[table[['A','B','C'].index(role)] for role in [numerator,denominator]]
                ratio=sum(n)/sum(d);halves=[sum(n[i:i+24])/sum(d[i:i+24]) for i in [0,24]]
                quarters=[sum(n[i:i+12])/sum(d[i:i+12]) for i in [0,12,24,36]]
                old=next(c for c in original[(job['case'],job['policy'])]['boundaries'][boundary]['controls'] if (c['numerator'],c['denominator'])==(numerator,denominator))
                assert math.isclose(ratio,old['visits'][job['visit']],rel_tol=1e-14)
                failed=not .99<=ratio<=1.01
                same_sign=all((h-1)*(ratio-1)>0 for h in halves)
                both_fail_same_direction=all(h>1.01 for h in halves) if ratio>1 else all(h<.99 for h in halves)
                pairs.append(dict(numerator=numerator,denominator=denominator,ratio=ratio,halves=halves,quarters=quarters,original_visit_failed=failed,
                    same_direction_both_halves=same_sign,failed_both_halves_same_direction=failed and both_fail_same_direction))
            cohorts.append(dict(job=job,boundary=boundary,frequency=members['A']['frequency'],calls_per_batch=members['A']['specification']['calls'],
                batch_tick_sums=table,decomposition=decomposition,pairs=pairs))
    assert counts==dict(measured=89088,managed_measured=66816,native_measured=22272,conditioning=227583,solo=5120)
    summary=[]
    for case in CASES:
        for policy in ['default','memory']:
            for boundary in ['execute','request']:
                rows=[r for r in cohorts if (r['job']['case'],r['job']['policy'],r['boundary'])==(case,policy,boundary)];assert len(rows)==4
                failed=[p for r in rows for p in r['pairs'] if p['original_visit_failed']]
                summary.append(dict(case=case,policy=policy,boundary=boundary,failed_pairs=len(failed),failed_pairs_same_sign=sum(p['same_direction_both_halves'] for p in failed),
                    failed_in_both_halves=sum(p['failed_both_halves_same_direction'] for p in failed),
                    process_share_median=statistics.median(r['decomposition']['shares']['process'] for r in rows),
                    process_share_range=[min(r['decomposition']['shares']['process'] for r in rows),max(r['decomposition']['shares']['process'] for r in rows)],
                    cycle_share_median=statistics.median(r['decomposition']['shares']['cycle'] for r in rows),
                    residual_share_median=statistics.median(r['decomposition']['shares']['residual'] for r in rows)))
    inputs[str((SOURCE/'aa-closed.json').relative_to(ROOT))]=pin(SOURCE/'aa-closed.json')
    inputs[str((SOURCE/'aa-final-verification.json').relative_to(ROOT))]=pin(SOURCE/'aa-final-verification.json')
    write(BASE/'analysis.json',dict(scope='Post-hoc balanced resident-process decomposition; no revised verdict, corrected timing, independence or causal claim.',counts=counts,inputs=inputs,workers=workers,cohorts=cohorts,summary=summary))
    print(json.dumps(dict(counts=counts,summary=[r for r in summary if r['boundary']=='execute'])))

if __name__=='__main__':main()
