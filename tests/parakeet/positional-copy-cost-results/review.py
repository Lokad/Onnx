"""Preserve the failed component gate and describe the complete observed spread."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-positional-copy-cost-amd-20260924'
OUT=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text(encoding='utf8'))
def pin(p):
    with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def main():
    closure=read(BASE/'closed.json');a=read(BASE/'analysis.json')
    assert pin(BASE/'closed.json')['sha256']=='c4c9aba12446a6216f35e213a5fcd1b3282d1a6261caab0f94029be81af1f8ef'
    assert closure['passed'] and closure['evidence_qualified'] and a['evidence_qualified']
    assert closure['analysis']==pin(BASE/'analysis.json') and closure['clocks']==pin(BASE/'clocks.json')
    assert not closure['component_stable'] and not closure['useful_component_estimate']
    failed=[r for r in a['controls'] if not r['passed']]
    assert len(failed)==13 and all(r['mode']=='helper' and r['name']!='corpus' for r in failed)
    assert all(r['passed'] for r in a['controls'] if r['mode']=='generic' or r['name']=='corpus')
    generic=[r for r in a['processes'] if r['mode']=='generic'];helper=[r for r in a['processes'] if r['mode']=='helper']
    casewise=sum(min(r['cases'][i]['seconds'] for r in generic)-max(r['cases'][i]['seconds'] for r in helper) for i in range(20))
    parts=defaultdict(int)
    for row in read(BASE/'clocks.json'):
        if row['pass']>0:parts[row['mode'],row['process'],row['pass'],row['case_index']]+=row['end_ticks']-row['start_ticks']
    spread=sum(min(v for (m,p,n,c),v in parts.items() if m=='generic' and c==i)
        -max(v for (m,p,n,c),v in parts.items() if m=='helper' and c==i) for i in range(20))/1e9
    absolute=[abs(helper[0]['cases'][i]['seconds']-helper[1]['cases'][i]['seconds']) for i in range(20)]
    value=dict(passed=True,analysis_only=True,original_comparison_admitted=False,
        original_verdict_unchanged=True,no_rerun=True,application_gain_claimed=False,
        closure=pin(BASE/'closed.json'),analysis=pin(BASE/'analysis.json'),clocks=pin(BASE/'clocks.json'),
        controls_passed=29,controls_total=42,failed_controls=failed,processes=a['processes'],
        corpus_seconds=a['corpus_seconds'],observed_difference_seconds=a['component_difference_seconds'],
        casewise_process_difference_seconds=casewise,casewise_all_measured_pass_difference_seconds=spread,
        largest_helper_case_difference_seconds=max(absolute),sum_helper_case_differences_seconds=sum(absolute),
        bounds_are_only_over_observed_samples=True,confidence_interval_claimed=False,
        interpretation='The original comparison fails 13 helper per-case controls and is not admitted. Both corpus controls and every generic-path control pass. Stable generic copying accounts for about 2.41 seconds. Even combining the fastest observed generic case/pass with the slowest helper case/pass leaves 2.020598 seconds in this finite sample. This supports one isolated copying prototype, not a speedup claim or relaxation of any original gate.',
        next_evaluation='One isolated reuse of the existing contiguous-copy helper; all numerical, full-application and release gates remain mandatory.',
        reviewer=pin(Path(__file__)))
    with (OUT/'observations-20260924.json').open('x',encoding='utf8') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(dict(report=pin(OUT/'observations-20260924.json'),original_comparison_admitted=False,
        observed_casewise_process_difference=casewise,observed_casewise_pass_difference=spread)))


if __name__=='__main__':main()
