"""Reconstruct the decomposition using independent rational means and raw calls."""
from fractions import Fraction
import json,math,statistics
from analyze import ROOT,SOURCE,BASE,pin,read,write
from test_analysis import independent

def main():
    value=read(BASE/'analysis.json')
    for name,wanted in value['inputs'].items():assert pin(ROOT/name)==wanted,name
    raw={};counts=dict(measured=0,managed_measured=0,native_measured=0,conditioning=0,solo=0)
    for worker in value['workers']:
        v=read(SOURCE/worker['source']);raw[(worker['cohort'],worker['role'])]=v
        assert worker['measured_calls']==len(v['measured']) and worker['conditioning_calls']==len(v['conditioning']) and worker['solo_calls']==len(v['solo'])
        counts['measured']+=len(v['measured']);counts['native_measured' if worker['role']=='N' else 'managed_measured']+=len(v['measured'])
        counts['conditioning']+=len(v['conditioning']);counts['solo']+=len(v['solo'])
        assert worker['gc']==[sum(r[k] for r in v['measured']) for k in ['g0','g1','g2']]
        assert worker['calls_with_gc']==sum(any(r[k] for k in ['g0','g1','g2']) for r in v['measured'])
        assert math.isclose(worker['mean_allocated'],float(Fraction(sum(r['bytes'] for r in v['measured']),len(v['measured']))),rel_tol=1e-14)
    assert len(raw)==160 and counts==value['counts']
    integer_checks=ratio_checks=0
    for cohort in value['cohorts']:
        boundary=cohort['boundary'];name=cohort['job']['name']
        table=[]
        for role in ['A','B','C']:
            v=raw[(name,role)]
            table.append([sum(r[boundary] for r in v['measured'] if r['block']==block) for block in range(48)])
        assert table==cohort['batch_tick_sums']
        expected=independent(table)
        for key,number in expected.items():
            assert Fraction(cohort['decomposition'][key],144**2)==number;integer_checks+=1
            if key!='total':assert math.isclose(float(number/expected['total']),cohort['decomposition']['shares'][key],rel_tol=1e-14)
        for pair in cohort['pairs']:
            n=raw[(name,pair['numerator'])]['measured'];d=raw[(name,pair['denominator'])]['measured']
            ratio=Fraction(sum(r[boundary] for r in n),sum(r[boundary] for r in d))
            assert float(ratio)==pair['ratio']
            assert pair['original_visit_failed']==(ratio<Fraction(99,100) or ratio>Fraction(101,100))
            ratios={}
            for label,width in [('halves',24),('quarters',12)]:
                ratios[label]=[Fraction(sum(r[boundary] for r in n if start<=r['block']<start+width),sum(r[boundary] for r in d if start<=r['block']<start+width)) for start in range(0,48,width)]
                assert [float(r) for r in ratios[label]]==pair[label];ratio_checks+=len(ratios[label])
            assert pair['same_direction_both_halves']==all((h-1)*(ratio-1)>0 for h in ratios['halves'])
            assert pair['failed_both_halves_same_direction']==(pair['original_visit_failed'] and
                (all(h>Fraction(101,100) for h in ratios['halves']) if ratio>1 else all(h<Fraction(99,100) for h in ratios['halves'])))
    for row in value['summary']:
        selected=[r for r in value['cohorts'] if (r['job']['case'],r['job']['policy'],r['boundary'])==(row['case'],row['policy'],row['boundary'])];assert len(selected)==4
        failed=[p for r in selected for p in r['pairs'] if p['original_visit_failed']]
        assert row['failed_pairs']==len(failed) and row['failed_pairs_same_sign']==sum(p['same_direction_both_halves'] for p in failed)
        assert row['failed_in_both_halves']==sum(p['failed_both_halves_same_direction'] for p in failed)
        for key in ['process','cycle','residual']:
            numbers=[r['decomposition']['shares'][key] for r in selected]
            assert row[key+'_share_median']==statistics.median(numbers)
            if key=='process':assert row['process_share_range']==[min(numbers),max(numbers)]
    assert integer_checks==320 and ratio_checks==1440 and len(value['summary'])==20
    write(BASE/'verification.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),inputs=len(value['inputs']),counts=counts,integer_component_checks=integer_checks,half_quarter_ratios=ratio_checks,summary_rows=20))
    print(json.dumps(read(BASE/'verification.json')))

if __name__=='__main__':main()
