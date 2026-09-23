"""Retain the pre-measurement consumer compile failure without a timing result."""
from pathlib import Path
import json,sys
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/benchmarks/release-amd'))
from protocol import pin,read,save,check_sample
BASE=ROOT/'artifacts/release-graph-baseline-amd-20260923'

def main():
    assert not (BASE/'closed.json').exists()
    c=BASE/'collected';receipt=read(c/'collection.json');state=read(c/'identity.json')
    assert receipt['terminal'] and receipt['code']==1 and receipt['input_error'] is None
    for name,wanted in receipt['files'].items():assert pin(c/name)==wanted,name
    assert state['complete'] and state['code']==1 and [r['name'] for r in state['runs']]==['sdk-version','consumer-restore','consumer-build']
    assert [r['code'] for r in state['runs']]==[0,0,1]
    log=(c/'logs/consumer-build.stdout').read_text()
    assert "Program.cs(40,20): error CS1061: 'ITensor' does not contain a definition for 'Dimensions'" in log
    assert "Program.cs(64,58): error CS1061: 'ITensor' does not contain a definition for 'Dimensions'" in log
    for row in state['runs']:
        samples=[json.loads(s) for s in (c/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0
        for sample in samples:check_sample(sample)
    save(BASE/'closed.json',dict(passed=False,retained_failure=True,no_numerical_or_timing_jobs=True,reason='Two consumer shape checks use typed Dimensions instead of public ITensor.Dims.',files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'))))

if __name__=='__main__':main()
