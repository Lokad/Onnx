"""Publish every counter block and cohort of the closed diagnostic, never a score."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-pad-memory-diagnostic-amd-20260926'


def pin(path):
    with path.open('rb') as f:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def flatten(row):
    value={}
    for key,item in row.items():
        if isinstance(item,dict):
            for stat,number in item.items():value[key+'_'+stat]=number
        else:value[key]=item
    return value


def main():
    closure=read(BASE/'closed.json');assert closure['passed'] and not closure['admitted']
    for name,wanted in closure['files'].items():assert pin(BASE/name)==wanted,name
    analysis=read(BASE/'analysis.json');assert analysis['diagnostic_only'] and not analysis['admitted']
    memory=analysis['memory'];assert memory['passed']
    for kind in ('blocks','calibrations'):
        values=[flatten(r) for r in memory[kind]]
        with (OUT/f'managed-{kind}-20260926.csv').open('x',encoding='utf8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]),lineterminator='\n');writer.writeheader();writer.writerows(values)
    cohorts=[]
    for case in memory['cases']:
        for key,group in case['cohorts'].items():
            cohorts.append(flatten(dict(process=case['process'],case=case['case'],cohort=key,**group)))
    with (OUT/'managed-cohorts-20260926.csv').open('x',encoding='utf8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(cohorts[0]),lineterminator='\n');writer.writeheader();writer.writerows(cohorts)
    summary=dict(closure=pin(BASE/'closed.json'),generator=pin(Path(__file__)),
        products=analysis['products'],calls=memory['calls'],calibration_calls=memory['calibration_calls'],
        cases=memory['cases'],calibrations=memory['calibrations'],prefixes=analysis['prefixes'],
        fixed_blocks=len(memory['blocks']),resources=analysis['resources'],peak_rss=analysis['peak_rss'],
        diagnostic_only=True,admitted=False)
    with (OUT/'managed-observations-20260926.json').open('x',encoding='utf8') as f:
        f.write(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(closure=summary['closure'],calls=memory['calls'],blocks=len(memory['blocks']),
                         cohorts=len(cohorts),resources=analysis['resources'],peak_rss=analysis['peak_rss'])))


if __name__=='__main__':main()
