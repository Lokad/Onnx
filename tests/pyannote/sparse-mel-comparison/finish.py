"""After actual application termination, audit, compare and close exactly once."""
import json
import subprocess
import time
import traceback
from common import *

FINISH = ROOT / 'artifacts/pyannote-sparse-mel-comparison-finish-20260921'
APPLICATION_ID = {'pid': 644148, 'birth': 1790026526.96885}
APPLICATION_TOOLS = ROOT / 'tests/pyannote/sparse-mel-qualification'


def live(identity):
    try:
        return psutil.Process(identity['pid']).create_time() == identity['birth']
    except psutil.NoSuchProcess:
        return False


def main():
    FINISH.mkdir()
    own = psutil.Process()
    original_affinity = own.cpu_affinity()
    own.cpu_affinity([0])
    files = {p.relative_to(ROOT).as_posix(): pin(p) for p in [*TOOLS.glob('*.py'), *APPLICATION_TOOLS.glob('*.py')]}
    save(FINISH / 'prepared.json', dict(application=APPLICATION_ID, tools=files,
        application_state=str((QUALIFIED / 'qualification.json').relative_to(ROOT)),
        stages=['audit-application', 'prepare-comparison', 'run-comparison', 'audit-comparison'],
        maximum_seconds=21600, scope='Wait for actual success and termination; no overlap, retries, threshold changes or VM work.'))
    state = dict(complete=False, code=None, phase='waiting-application',
        supervisor=dict(pid=own.pid, birth=own.create_time()), stages=[])
    path = FINISH / 'state.json'
    save(path, state)
    started = time.monotonic()

    def stage(name, tool, arguments, seconds):
        verify(files)
        row = dict(name=name, complete=False, code=None, members={}, started=time.time())
        state['stages'].append(row)
        state['phase'] = name
        save(path, state)
        child = None
        beginning = time.monotonic()
        try:
            with (FINISH / (name + '.log')).open('x', encoding='utf8') as log:
                child = subprocess.Popen([sys.executable, '-X', 'utf8', '-B', str(tool), *arguments], cwd=ROOT,
                    env=clean_env(), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW)
                process = psutil.Process(child.pid)
                row['worker'] = dict(pid=process.pid, birth=process.create_time())
                row['members'][str(process.pid)] = process.create_time()
                save(path, state)
                while child.poll() is None:
                    for member in [process, *process.children(recursive=True)]:
                        try:
                            birth = member.create_time()
                            assert row['members'].get(str(member.pid), birth) == birth
                            row['members'][str(member.pid)] = birth
                        except psutil.NoSuchProcess:
                            pass
                    row['seconds'] = time.monotonic() - beginning
                    save(path, state)
                    assert row['seconds'] < seconds and time.monotonic() - started < 21600
                    time.sleep(1)
                row['code'] = child.wait()
                assert row['code'] == 0, (name, row['code'])
                for pid, birth in row['members'].items():
                    terminal(dict(pid=int(pid), birth=birth))
        except BaseException:
            row['error'] = traceback.format_exc()
            for pid, birth in reversed(list(row['members'].items())):
                identity = dict(pid=int(pid), birth=birth)
                if live(identity):
                    psutil.Process(identity['pid']).kill()
            if child is not None:
                child.wait(timeout=15)
            raise
        finally:
            row.update(complete=True, ended=time.time(), seconds=time.monotonic()-beginning)
            save(path, state)

    try:
        while True:
            observed = read(QUALIFIED / 'qualification.json')
            assert observed['supervisor'] == APPLICATION_ID
            state['application_complete'] = observed['complete']
            state['application_live'] = live(APPLICATION_ID)
            state['wait_seconds'] = time.monotonic() - started
            save(path, state)
            assert state['wait_seconds'] < 5400, 'Waiting ceiling; do not restart the application'
            if observed['complete'] and not state['application_live']:
                assert observed['code'] == 0, 'Application qualification failed; no comparison admission'
                for run in observed['runs']:
                    assert run['complete'] and run['code'] == 0
                    for pid, birth in run['members'].items():
                        terminal(dict(pid=int(pid), birth=birth))
                break
            assert observed['complete'] or state['application_live'], 'Incomplete application controller is absent'
            time.sleep(15)
        stage('audit-application', APPLICATION_TOOLS / 'audit.py', [], 300)
        closure = read(QUALIFIED / 'closed.json')
        assert closure['passed']  # Allocation reduction is not a computation-admission requirement.
        stage('prepare-comparison', TOOLS / 'prepare.py', [pin(QUALIFIED / 'closed.json')['sha256']], 300)
        stage('run-comparison', TOOLS / 'run.py', [], 12000)
        stage('audit-comparison', TOOLS / 'audit.py', [], 300)
        state.update(code=0, phase='complete', application_closure=pin(QUALIFIED / 'closed.json'),
            comparison_closure=pin(BASE / 'closed.json'), attribution_valid=read(BASE / 'analysis.json')['attribution_valid'])
    except BaseException:
        state.update(code=1, error=traceback.format_exc())
        raise
    finally:
        state['complete'] = True
        save(path, state)
        own.cpu_affinity(original_affinity)


if __name__ == '__main__':
    main()
