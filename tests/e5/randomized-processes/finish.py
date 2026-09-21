"""Continue the fixed e5 campaign once Whisper is closed; never retry inference."""
import datetime
import json
import os
import subprocess
import time
import traceback

from contract import pin, read, write
from remote import BASE, ROOT, MONITOR, WHISPER, SITE, sys

CONTROL = BASE.with_name('e5-randomized-processes-finish-20260921')


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf8')
    temporary.replace(path)


def main(failed_predecessor_sha256=None):
    global CONTROL
    if failed_predecessor_sha256:
        CONTROL = BASE.with_name('e5-randomized-processes-finish-v2-20260921')
    assert not CONTROL.exists() and not (BASE/'deployment-aa.json').exists()
    prepared = pin(BASE/'prepared.json'); assert read(BASE/'prepared.json')['passed'] is True
    sys.path.insert(0, str(SITE)); import psutil
    own = psutil.Process(); own.cpu_affinity([0])
    folder = ROOT/'tests/e5/randomized-processes'
    paths = list(folder.glob('*.py'))+[ROOT/'tests/e5/process-uncertainty'/name for name in
            ['estimator.py', 'protocol.py', 'worker_audit.py']]+[ROOT/'eng/campaign_processes.py']
    tools = {p.relative_to(ROOT).as_posix(): pin(p) for p in paths}
    CONTROL.mkdir(); state_path = CONTROL/'state.json'
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()),
                 prepared=prepared, tools=tools, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 phase='waiting-for-whisper', stages=[], observations=0)
    write(state_path, state)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    started = time.monotonic(); child = None

    def live(identity):
        try:
            return psutil.Process(identity['pid']).create_time() == identity['birth']
        except psutil.NoSuchProcess:
            return False

    def unchanged():
        assert pin(BASE/'prepared.json') == prepared
        for name, wanted in tools.items():
            assert pin(ROOT/name) == wanted, ('Controller source changed', name)

    def step(name, phase=None, extra=()):
        nonlocal child
        unchanged()
        label = name+('' if phase is None else '-'+phase)
        stage = dict(name=label, complete=False, code=None, started=time.time())
        state['stages'].append(stage); save(state_path, state)
        command = [sys.executable, '-X', 'utf8', '-B', str(folder/name)]
        if phase is not None:
            command += ['--phase', phase]
        command += list(extra)
        with (CONTROL/(label+'.stdout')).open('x') as out, (CONTROL/(label+'.stderr')).open('x') as err:
            child = subprocess.Popen(command, env=env, stdout=out, stderr=err, creationflags=subprocess.CREATE_NO_WINDOW)
            stage['child'] = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time()); save(state_path, state)
            # A timeout stops this controller. It never restarts or kills the
            # reporting child or any remote workload; inspect actual births.
            stage['code'] = child.wait(timeout=7200)
        stage.update(complete=True, ended=time.time()); save(state_path, state)
        assert stage['code'] == 0, (label, stage['code'])
        child = None
        print(json.dumps(dict(stage=label, code=0)), flush=True)

    def observe_phase(phase):
        phase_started = time.monotonic()
        with (CONTROL/('observations-'+phase+'.jsonl')).open('x', encoding='utf8') as observations:
            while True:
                assert time.monotonic()-phase_started < 32*3600, 'Observation ceiling: inspect existing remote processes'
                unchanged()
                try:
                    result = subprocess.run([sys.executable, '-X', 'utf8', '-B', str(folder/'observe.py'), '--phase', phase],
                              env=env, capture_output=True, text=True, encoding='utf8', timeout=60,
                              creationflags=subprocess.CREATE_NO_WINDOW)
                    event = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                 code=result.returncode, stdout=result.stdout, stderr=result.stderr)
                except subprocess.TimeoutExpired:
                    event = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), code=None, timeout=True)
                observations.write(json.dumps(event)+'\n'); observations.flush()
                state['observations'] += 1
                if event['code'] == 0:
                    value = json.loads(event['stdout']); state['last_observation'] = value
                    save(state_path, state)
                    if not value['supervisor_live']:
                        if value.get('initializing'):
                            raise RuntimeError('Supervisor stopped before identity creation; retain stderr and inspect, do not relaunch')
                        latest = value['latest']
                        if latest is None or all(not b['live'] for b in latest['births']):
                            assert value['complete'] is True, 'Terminal supervisor left incomplete identity'
                            return value
                else:
                    state['last_observation_error'] = event; save(state_path, state)
                time.sleep(30)

    try:
        while True:
            assert time.monotonic()-started < 5*3600, 'Whisper wait ceiling; inspect its original monitor'
            prior = read(MONITOR/'state.json')
            if prior['complete']:
                if failed_predecessor_sha256:
                    closure = WHISPER/'failure-closed-v2.json'
                    assert pin(closure)['sha256'] == failed_predecessor_sha256 and prior['code'] == 1
                    failure = read(closure)
                    assert failure['closure_passed'] is True and failure['campaign_passed'] is False
                else:
                    assert prior['code'] == 0, 'Whisper completion failed; no e5 launch'
                if not live(prior['supervisor']):
                    if not failed_predecessor_sha256:
                        assert read(WHISPER/'final-verification.json')['passed'] is True
                    break
            else:
                assert live(prior['supervisor']), 'Whisper monitor disappeared; inspect, do not duplicate reporting'
            time.sleep(30)
        unchanged(); step('stage.py', extra=('--failed-predecessor-closure', failed_predecessor_sha256) if failed_predecessor_sha256 else ())
        for phase in ['aa', 'compare']:
            state['phase'] = phase; save(state_path, state)
            if phase == 'compare':
                step('start_compare.py')
            terminal = observe_phase(phase)
            step('collect.py', phase)
            assert terminal['code'] == 0, 'Failed inference collected without scoring or retry'
            for name in ['report.py', 'verify_report.py', 'update_benchmark.py']:
                step(name, phase)
            verified = read(BASE/(phase+'-verification.json'))
            assert verified['passed'] is True
            if phase == 'aa' and not (verified['statistical_screen'] and verified['diagnostic_screen']):
                state.update(code=0, outcome='A/A did not pass; full report retained, comparison not launched')
                break
        else:
            state.update(code=0, outcome='Both fixed phases fully collected and independently verified')
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        if child is not None:
            state['last_child_still_live'] = child.poll() is None
        raise
    finally:
        state.update(complete=True, ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                     seconds=time.monotonic()-started)
        save(state_path, state)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--failed-predecessor-closure')
    main(parser.parse_args().failed_predecessor_closure)
