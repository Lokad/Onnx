"""Wait for the existing e5 owner, then execute and collect one audio campaign."""
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
from candidate_protocol import pin, read, write
from transport import BASE, ROOT, TOOLS, SITE, E5_CONTROL, checked_local, local_e5_terminal, observe


def main():
    sys.path.insert(0, str(SITE)); import psutil
    checked_local()
    folder = BASE/'controller'; folder.mkdir()
    own = psutil.Process(); own.cpu_affinity([0])
    state = dict(complete=False, code=None, phase='waiting-for-e5', started=time.time(),
                 supervisor=dict(pid=own.pid, birth=own.create_time()), prepared=pin(BASE/'prepared.json'), stages=[])
    def save():
        path = folder/'state.tmp'; path.write_text(json.dumps(state, indent=2), encoding='utf8'); path.replace(folder/'state.json')
    save(); started = time.monotonic()
    def step(script, *args):
        checked_local()
        label = Path(script).stem+('-'+args[0] if args else '')
        row = dict(name=label, complete=False, code=None, started=time.time()); state['stages'].append(row); save()
        with (folder/(label+'.stdout')).open('x') as out, (folder/(label+'.stderr')).open('x') as err:
            child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(TOOLS/script), *args], cwd=ROOT,
                                     stdout=out, stderr=err, stdin=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            row['child'] = dict(pid=child.pid, birth=psutil.Process(child.pid).create_time()); save()
            try:
                row['code'] = child.wait(timeout=1800)
            finally:
                row.update(complete=child.poll() is not None, ended=time.time()); save()
        assert row['code'] == 0, (label, row['code'])
    try:
        while True:
            assert time.monotonic()-started < 72*3600, 'Wait ceiling; inspect actual e5 owner without relaunching'
            prior = read(E5_CONTROL)
            if prior['complete']:
                # The receipt is written just before the controller exits. Give
                # that final writer (and any collecting child) time to finish.
                identities = [prior['supervisor']]+[s['child'] for s in prior['stages'] if 'child' in s]
                active = []
                for identity in identities:
                    try:
                        if psutil.Process(identity['pid']).create_time() == identity['birth']: active.append(identity)
                    except psutil.NoSuchProcess:
                        pass
                if not active: break
            else:
                identity = prior['supervisor']
                assert psutil.Process(identity['pid']).create_time() == identity['birth'], 'e5 controller vanished; inspect existing workers'
            time.sleep(30)
        local_e5_terminal()
        state['phase'] = 'staging'; save(); step('transport.py', 'stage')
        state['phase'] = 'launching'; save(); step('transport.py', 'launch')
        state['phase'] = 'observing'; save(); observing = time.monotonic()
        with (folder/'observations.jsonl').open('x') as log:
            while True:
                assert time.monotonic()-observing < 5*3600, 'Observation ceiling; inspect actual worker, never duplicate it'
                try:
                    value = observe(); event = dict(time=time.time(), observation=value)
                except (subprocess.TimeoutExpired, AssertionError) as error:
                    value = None; event = dict(time=time.time(), observation_error=repr(error))
                log.write(json.dumps(event)+'\n'); log.flush(); state['last_observation'] = event; save()
                if value is not None and not value['supervisor_live'] and not value['live']:
                    break
                time.sleep(30)
        state['phase'] = 'collecting'; save(); step('transport.py', 'collect')
        state['phase'] = 'auditing'; save(); step('audit_results.py')
        state.update(code=0, phase='collected-and-audited', outcome='AMD evidence audited; publish report and decide candidate separately')
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); traceback.print_exc()
    finally:
        state.update(complete=True, ended=time.time(), seconds=time.monotonic()-started); save()
    return state['code']


if __name__ == '__main__':
    raise SystemExit(main())
