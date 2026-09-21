"""Verify completed runner smokes and its terminal failure path without inference."""
from pathlib import Path
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'artifacts/e5-randomized-processes-20260921'
SITE = ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'
sys.path.append(str(SITE))
import psutil
from contract import audit, read, write, pin
from run import terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-after-negative', action='store_true')
    args = parser.parse_args()
    assert not (BASE/'runner-verification.json').exists()
    state = read(BASE/'local-state.json')
    assert state['complete'] is True and state['code'] == 0 and state['manifest_refusals'] == 10
    terminal([state['supervisor']]+state['children'])
    base = BASE/'local-payload'
    all_births = [state['supervisor']]
    for phase in ('aa', 'compare'):
        result = audit(base, phase)
        assert result == read(BASE/('local-'+phase+'-audit.json'))
        terminal(result['resources']['births']); all_births.extend(result['resources']['births'])
    # The completed success smoke predates exactly two narrowly scoped fixes:
    # an analysis-only diagnostic and marking the runner's failure terminal.
    old_contract = (base/'tests/e5/randomized-processes/contract.py').read_text()
    current_contract = Path(__file__).with_name('contract.py').read_text()
    old = "                    differences = [a-Fraction(result['ratio'])*b for a, b in zip(y, x, strict=True)]"
    new = "                    fitted_ratio = sum(y)/sum(x)\n                    differences = [a-fitted_ratio*b for a, b in zip(y, x, strict=True)]"
    assert old_contract.count(old) == 1 and old_contract.replace(old, new) == current_contract
    old_runner = (base/'tests/e5/randomized-processes/run.py').read_text()
    current_runner = Path(__file__).with_name('run.py').read_text()
    old = '        state.update(code=2, error=traceback.format_exc()); raise'
    new = '        state.update(complete=True, code=2, error=traceback.format_exc()); raise'
    assert old_runner.count(old) == 1 and old_runner.replace(old, new) == current_runner

    bad = BASE/'negative-payload'
    if args.resume_after_negative:
        assert bad.exists() and (BASE/'verification-recovery.json').exists()
        identity = read(BASE/'negative-launch.json')['child']
    else:
        bad.mkdir(exist_ok=False)
        meta = read(base/'frozen.json')
        for name in meta['files']:
            target = bad/name; target.parent.mkdir(parents=True, exist_ok=True)
            if name.endswith(('/contract.py', '/run.py')):
                shutil.copyfile(Path(__file__).with_name(Path(name).name), target)
            else:
                os.link(base/name, target)
        meta['files'] = {name: pin(bad/name) for name in meta['files']}
        # The worker rejects this SHA against its fixed model fixture before model
        # import, native session creation or inference. This exercises real startup
        # and owned-process cleanup without deliberately doing invalid inference.
        wrong_model = ROOT/'global.json'
        meta['model'] = dict(path=wrong_model.as_posix(), **pin(wrong_model))
        write(bad/'frozen.json', meta)
        env = dict(os.environ, PYTHONPATH=str(SITE), PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
        with (BASE/'negative.stdout').open('x') as stdout, (BASE/'negative.stderr').open('x') as stderr:
            child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(bad/'tests/e5/randomized-processes/run.py'),
                                      '--payload', str(bad), '--phase', 'aa'], env=env, stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW)
            identity = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time())
            write(BASE/'negative-launch.json', dict(child=identity, started=time.time(), frozen=pin(bad/'frozen.json')))
            try:
                code = child.wait(timeout=60)
            except BaseException:
                try:
                    process = psutil.Process(identity['pid'])
                    if process.create_time() == identity['birth']:
                        for member in reversed(process.children(recursive=True)):
                            member.kill()
                        process.kill()
                except psutil.NoSuchProcess:
                    pass
                child.wait(timeout=10)
                raise
        assert code != 0
    terminal([identity])
    failure = read(bad/'result-aa/identity.json')
    assert failure['complete'] is True and failure['code'] == 2 and len(failure['runs']) == 1
    run = failure['runs'][0]
    terminal([dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items()])
    directory = bad/'result-aa'/run['job']['name']
    assert 'Model identity' in (directory/'stderr.txt').read_text()
    assert not (directory/'output/result.json').exists()
    assert 'AssertionError' in failure['error']
    all_births += [identity]+[dict(pid=int(pid), birth=birth) for pid, birth in run['members'].items()]
    tests = (BASE/'tests-final.log').read_text(encoding='utf-8-sig')
    assert 'Ran 9 tests' in tests and '\nOK' in tests
    output = dict(passed=True, positive_workers=12, measured_calls=48, conditioning_calls=80,
                  manifest_refusals=10, negative_workers=1, failure_before_inference=True,
                  birth_count=len(all_births), tests=9,
                  source_changes=dict(only_exact_variance_diagnostic_and_terminal_failure_marker=True,
                                      contract=pin(Path(__file__).with_name('contract.py')), runner=pin(Path(__file__).with_name('run.py'))),
                  files={p.relative_to(BASE).as_posix(): pin(p) for p in sorted(BASE.rglob('*')) if p.is_file()},
                  verifier=pin(Path(__file__)))
    write(BASE/'runner-verification.json', output)
    print(json.dumps({k: v for k, v in output.items() if k not in ('files', 'verifier', 'source_changes')}))


if __name__ == '__main__':
    main()
