"""Check real-record refusal behavior and independently reproduce process accounting."""
from pathlib import Path
import argparse
import copy
import importlib.util
import math
from audit import validate_records
from common import read,sha,write_new


def check(base):
    audit=read(base/'audit.json');assert audit['execution_passed']
    identity=read(base/'run/identity.json');audio=read(base/'inputs/audio.json')
    spec=importlib.util.spec_from_file_location('frozen_accounting',base/'runtime-source/campaign_processes.py')
    accounting=importlib.util.module_from_spec(spec);spec.loader.exec_module(accounting)
    mutations={
        'missing_record':lambda r:r.pop(),
        'changed_case_name':lambda r:r[3].__setitem__('name','wrong'),
        'changed_language':lambda r:r[3].__setitem__('language','unknown'),
        'changed_pcm_identity':lambda r:r[3].__setitem__('pcm_sha256','0'*64),
        'invalid_elapsed':lambda r:r[3].__setitem__('seconds',float('nan')),
        'invalid_clock':lambda r:r[3].__setitem__('end_ticks',r[3]['start_ticks']),
        'changed_ownership':lambda r:r[3].__setitem__('input_and_held_results_unchanged',False),
        'changed_repeat':lambda r:r[-1]['decision'].__setitem__('text','corrupted repeat'),
        'invalid_token_type':lambda r:r[3]['decision'].__setitem__('token_ids',[True]),
        'invalid_stop':lambda r:r[3]['decision'].__setitem__('stop_reason','unknown')}
    refusals=[];observed=[]
    for run in identity['runs']:
        family,engine=run['family'],run['engine'];path=base/'run'/run['name']/'result.json'
        original=read(path)['cases'];validate_records(family,engine,original,audio['cases'])
        for name,mutate in mutations.items():
            rows=copy.deepcopy(original);mutate(rows)
            try:validate_records(family,engine,rows,audio['cases'])
            except (AssertionError,KeyError,ValueError):refusals.append(dict(worker=run['name'],mutation=name))
            else:raise AssertionError(('Damaged real record accepted',run['name'],name))
        before=read(base/'run'/(run['name']+'-pre.json'));after=read(base/'run'/(run['name']+'-post.json'))
        actual=accounting.foreign_fraction(before,after,identity['supervisor']['pid'])
        assert actual==run['accounting']
        observed.append(dict(worker=run['name'],result_sha256=sha(path),accounting=actual))
    # Recompute every group directly from its retained case counts.
    for model in audit['models'].values():
        for group in model['groups']:
            cases=[r for r in model['cases'] if r['condition']==group['condition'] and (group['locale']=='all' or r['locale']==group['locale'])]
            assert len(cases)==group['recordings']
            for engine in ('native','managed'):
                values=[r[engine+'_metrics'] for r in cases];actual=group[engine]
                for field in ('reference_words','reference_characters','word_errors','character_errors','substitutions','deletions','insertions'):
                    assert actual[field]==sum(v[field] for v in values)
                assert actual['word_error_rate']==actual['word_errors']/actual['reference_words']
                assert actual['character_error_rate']==actual['character_errors']/actual['reference_characters']
                assert math.isfinite(actual['word_error_rate']) and math.isfinite(actual['character_error_rate'])
    return dict(schema=1,passed=True,refusals=refusals,observed=observed,audit_sha256=sha(base/'audit.json'),checker_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();assert not args.output.exists()
    result=check(args.artifact.resolve());write_new(args.output,result)
    print('Rejected',len(result['refusals']),'damaged real records; process accounting and score arithmetic verified.')
