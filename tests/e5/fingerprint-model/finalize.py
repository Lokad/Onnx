"""Close a failed A/A or completed comparison without changing any timing criterion."""
from pathlib import Path
import argparse,json,shutil,time
import audit as checks
from close_phase import terminal

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    assert not (base/'closed.json').exists()
    aa=checks.read(base/'aa-closed.json');phases=['aa']
    if aa['timing_passed']:
        assert (base/'compare-closed.json').exists(),'Passing A/A requires completing the already authorized comparison'
        phases.append('compare')
    else:assert not (base/'deployment-compare.json').exists() and not (base/'aa-gate.json').exists()
    births=[]
    for phase in phases:
        receipt=checks.read(base/(phase+'-closed.json'))
        assert receipt['evidence_passed'] is True
        for name,wanted in receipt['evidence_files'].items():assert checks.pin(base/name)==wanted,name
        audit=checks.read(base/(phase+'-audit.json'));assert receipt['audit']==checks.pin(base/(phase+'-audit.json'))
        assert audit['passed'] is True and receipt['timing_passed']==audit['timing']['passed']
        births+=receipt['terminal']['births']
        label='aa' if phase=='aa' else 'comparison';source=Path(__file__).resolve().parent
        observations=checks.read(source/(label+'-observations-20260920.json'))
        assert observations['audit']==audit and observations['receipt']==checks.pin(base/(phase+'-closed.json'))
    verified=terminal(births)
    source=Path(__file__).resolve().parent
    shutil.copyfile(Path(__file__),base/'finalize-source.py')
    checks.write(base/'final-verification.json',dict(passed=True,terminal=verified,phases=phases,
        reports={name:checks.pin(source/name) for phase in phases for name in [(('aa' if phase=='aa' else 'comparison')+'-results-20260920.md'),(('aa' if phase=='aa' else 'comparison')+'-observations-20260920.json')]}))
    files={path.relative_to(base).as_posix():checks.pin(path) for path in sorted(base.rglob('*')) if path.is_file()}
    verdict='A/A failed; no measured candidate comparison' if not aa['timing_passed'] else 'Comparison '+('passed' if checks.read(base/'compare-closed.json')['timing_passed'] else 'failed or inconclusive')
    checks.write(base/'closed.json',dict(closed_at=time.time(),verdict=verdict,phases=phases,files=files))
    print(json.dumps(dict(closed=checks.pin(base/'closed.json'),files=len(files),verdict=verdict)))

if __name__=='__main__':main()
