"""Run one bounded offline build, AMD qualification and matched pyannote campaign."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from terminal_snapshot import observe as terminal_snapshot
from candidate_protocol import LIMITS, ROLES, TIMING_ROLES, REQUIRED_TESTS, gate, pin, read, write, verify, check_sample, test_results

DOTNET = '/home/vermorel/.dotnet/dotnet'


def build_prerequisites(base, worker, flags):
    projects = [('backend', base/'source/tests/Lokad.Onnx.Backend.Tests/Lokad.Onnx.Backend.Tests.csproj'),
                ('tensors', base/'source/tests/Lokad.Onnx.Tensors.Tests/Lokad.Onnx.Tensors.Tests.csproj'),
                ('cli', base/'source/src/Lokad.Onnx.CLI/Lokad.Onnx.CLI.csproj'),
                ('il-bridge', base/'il-bridge/IlBridge.csproj')]
    for name, project in projects:
        worker(name+'-restore', [DOTNET, 'restore', project, *flags, '--source', base/'nuget-feed',
                                '--packages', base/'work/packages', '--no-http-cache', '--disable-parallel', '-p:NuGetAudit=false'], build=True)
        worker(name+'-build', [DOTNET, 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers'], build=True)
    cli = base/'source/src/Lokad.Onnx.CLI/bin/Release/net10.0'
    for name in ('Lokad.Onnx.CLI.dll', 'Lokad.Onnx.CLI.deps.json', 'Lokad.Onnx.CLI.runtimeconfig.json'):
        assert (cli/name).is_file(), ('Missing Release CLI test prerequisite', name)
    return projects


def absent(identity):
    try:
        process = psutil.Process(identity['pid'])
        return process.create_time() != identity['birth'] or process.status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def e5_terminal():
    assert absent(dict(pid=673291, birth=1790056323.42)), 'M10 owner live'
    assert absent(dict(pid=679833, birth=1790058842.37)), 'Single-panel component owner live'
    assert absent(dict(pid=652149, birth=1790038671.33)), 'Previous combined owner live'
    base = Path('/dev/shm/lokad-e5-independent-20260921')
    assert absent(dict(pid=643872, birth=1790034498.73)), 'Previous audio supervisor live'
    if not base.exists():
        assert absent(dict(pid=408655, birth=1789962877.83)), 'Previous e5 supervisor live'
        for process in psutil.process_iter(['name', 'cmdline']):
            try:
                command = ' '.join(process.info['cmdline'] or [])
                assert process.info['name'] != 'dotnet', ('Unexpected dotnet owner', process.pid)
                assert 'lokad-e5-independent-20260921' not in command and 'lokad-pyannote-candidates-20260921' not in command, ('Previous worker live', process.pid)
            except psutil.NoSuchProcess:
                pass
        return dict(temporary_directories_cleared=True,
            collected_e5='2a6dde4b379a96a47c3b3a20e0d8e0057cd1db043de73808350a95538b1e766b',
            closed_audio='e0f1123a85c2a575e062fbf0e4dacc8c742c087661c2599d0197e5ca35cecc68')
    receipts = {}
    assert (base/'deployment-aa.json').is_file()
    for phase in ('aa', 'compare'):
        deployment = base/('deployment-'+phase+'.json')
        if not deployment.exists():
            continue
        assert absent(read(deployment)), ('Live e5 owner', phase)
        path = base/('result-'+phase)/'identity.json'; state = read(path)
        assert state['complete'] is True
        for run in state['runs']:
            worker = read(path.parent/run['job']['name']/'identity.json')
            assert 'ended' in worker, ('Incomplete e5 worker receipt', phase, run['job']['name'])
            for pid, birth in worker['members'].items():
                assert absent(dict(pid=int(pid), birth=birth)), ('Live e5 worker', phase, pid)
        receipts[phase] = dict(deployment=pin(deployment), identity=pin(path), code=state['code'])
    return receipts


def native_result(base, output, manifest_path, mode, spec):
    from protocol import validate_records
    result = read(output/'result.json'); manifest = read(manifest_path)
    validate_records(result, manifest, mode)
    assert result['manifest_sha256'] == pin(manifest_path)['sha256']
    assert result['engine'] == 'ort' and result['python_binary'] == spec['interpreter']
    assert result['runner_sha256'] == pin(base/'runtime/native.py')['sha256']
    assert result['versions'] == manifest['native_versions'] and result['native_binaries'] == manifest['native_binaries']
    assert result['native_settings'] == dict(provider='CPUExecutionProvider', intra_threads=1, inter_threads=1,
                                              sequential=True, graph_optimizations='all', spinning=False)
    assert result['numeric_libraries']
    for name, wanted in result['numeric_libraries'].items():
        assert pin(Path(name)) == spec['external'][name] == wanted, name
    return result


def run(base):
    assert os.name == 'posix' and psutil.__version__ == '7.0.0' and not sys.flags.optimize
    own = psutil.Process(); own.cpu_affinity([0])
    assert not (base/'campaign').exists()
    spec, execution = verify(base)
    assert pin(Path(sys.executable)) == spec['interpreter']
    e5 = e5_terminal()
    sys.path.insert(0, str(base/'runtime'))
    from protocol import validate_records
    from qualify_outputs import pyannote, parakeet
    accounting_spec = importlib.util.spec_from_file_location('accounting', base/'runtime/campaign_processes.py')
    account = importlib.util.module_from_spec(accounting_spec); accounting_spec.loader.exec_module(account)
    campaign = base/'campaign'; campaign.mkdir()
    state = dict(complete=False, code=None, started=time.time(), supervisor=dict(pid=own.pid, birth=own.create_time()),
                 payload=pin(base/'payload.json'), execution=pin(base/'execution/execution.json'), limits=LIMITS,
                 e5_terminal=e5, runs=[])
    def save():
        path = campaign/'identity.tmp'; path.write_text(json.dumps(state, indent=2), encoding='utf8')
        path.replace(campaign/'identity.json')
    env = {k: v for k, v in os.environ.items() if not k.lower().startswith(('lokad_', 'dotnet_', 'complus_'))}
    env.pop('PYTHONOPTIMIZE', None)
    env.update(PYTHONPATH=os.pathsep.join(spec['python_paths']), PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')
    env.update({k: '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'BLIS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')})
    for name in ('tmp', 'packages', 'cli-home', 'cache', 'nuget-http-cache'):
        (base/'work'/name).mkdir(parents=True)
    # Only scratch/cache locations change for model execution; no runtime knobs.
    env.update(TMPDIR=str(base/'work/tmp'), XDG_CACHE_HOME=str(base/'work/cache'))
    build_env = dict(env, DOTNET_CLI_HOME=str(base/'work/cli-home'),
                     DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1', DOTNET_CLI_TELEMETRY_OPTOUT='1',
                     NUGET_PACKAGES=str(base/'work/packages'), NUGET_HTTP_CACHE_PATH=str(base/'work/nuget-http-cache'),
                     MSBUILDDISABLENODEREUSE='1', DOTNET_CLI_USE_MSBUILD_SERVER='0')
    started = time.monotonic(); save()

    def worker(name, command, build=False, allowed=(0,)):
        assert time.monotonic()-started < LIMITS['campaign_seconds']
        assert psutil.virtual_memory().available >= LIMITS['preflight_available']
        assert psutil.disk_usage(str(base)).free >= LIMITS['preflight_tmpfs_free']
        folder = campaign/name; folder.mkdir()
        row = dict(name=name, command=list(map(str, command)), complete=False, code=None, started=time.time(),
                   preflight_available=psutil.virtual_memory().available, preflight_tmpfs_free=psutil.disk_usage(str(base)).free,
                   samples=0, peak_rss=0, members={})
        state['runs'].append(row); save(); child = None
        start = time.monotonic(); before = account.snapshot(); write(folder/'pre.json', before)
        last_size = 0.; artifact_bytes = 0
        try:
            with (folder/'stdout.txt').open('x') as out, (folder/'stderr.txt').open('x') as err, (folder/'samples.jsonl').open('x') as stream:
                own.cpu_affinity([2])
                try:
                    child = subprocess.Popen(row['command'], cwd=base/'source', env=build_env if build else env,
                                             stdin=subprocess.DEVNULL, stdout=out, stderr=err, start_new_session=True)
                finally:
                    own.cpu_affinity([0])
                process = psutil.Process(child.pid); row['child'] = dict(pid=child.pid, birth=process.create_time())
                row['members'][str(child.pid)] = row['child']['birth']; save()
                while child.poll() is None:
                    members = []
                    try:
                        assert process.create_time() == row['child']['birth']
                        for p in [process]+process.children(recursive=True):
                            try:
                                birth = p.create_time(); assert row['members'].get(str(p.pid), birth) == birth
                                row['members'][str(p.pid)] = birth
                                threads = []
                                for thread in p.threads():
                                    try:
                                        threads.append(dict(tid=thread.id, affinity=sorted(os.sched_getaffinity(thread.id))))
                                    except ProcessLookupError:
                                        pass
                                if p.status() == psutil.STATUS_ZOMBIE:
                                    continue
                                members.append(dict(pid=p.pid, birth=birth, rss=p.memory_info().rss,
                                                    affinity=p.cpu_affinity(), threads=threads))
                            except psutil.NoSuchProcess:
                                pass
                    except psutil.NoSuchProcess:
                        pass
                    if time.monotonic()-last_size >= 5:
                        artifact_bytes = 0
                        for path in base.rglob('*'):
                            try:
                                if path.is_file(): artifact_bytes += path.stat().st_size
                            except FileNotFoundError:
                                pass  # Build tools may atomically replace temporary files.
                        last_size = time.monotonic()
                    sample = dict(seconds=time.monotonic()-start, members=members, available=psutil.virtual_memory().available,
                                  tmpfs_free=psutil.disk_usage(str(base)).free, artifact_bytes=artifact_bytes)
                    transition = terminal_snapshot(child, members, row['members'], absent,
                        sample['seconds'], sample['available'], sample['tmpfs_free'], sample['artifact_bytes'])
                    if transition is not None:
                        row['terminal_transition'] = transition
                        write(folder/'terminal-transition.json', transition)
                        save()
                        break
                    stream.write(json.dumps(sample)+'\n'); stream.flush()
                    row['samples'] += 1; row['peak_rss'] = max(row['peak_rss'], sum(m['rss'] for m in members)); save()
                    check_sample(sample)
                    assert time.monotonic()-started < LIMITS['campaign_seconds']
                    time.sleep(.5)
                row['code'] = child.wait()
                assert row['code'] in allowed, (name, row['code'])
            deadline = time.monotonic()+15
            while not all(absent(dict(pid=int(pid), birth=birth)) for pid, birth in row['members'].items()):
                assert time.monotonic() < deadline, ('Unterminated owned child', name)
                time.sleep(.1)
            after = account.snapshot(); write(folder/'post.json', after)
            row['accounting'] = account.foreign_fraction(before, after, own.pid)
        except BaseException:
            row['error'] = traceback.format_exc()
            # Only recorded descendants with the same birth identity may be stopped.
            for pid, birth in reversed(list(row['members'].items())):
                if not absent(dict(pid=int(pid), birth=birth)):
                    try:
                        psutil.Process(int(pid)).kill()
                    except psutil.NoSuchProcess:
                        pass
            if child is not None:
                child.wait(timeout=15)
            raise
        finally:
            row.update(complete=True, seconds=time.monotonic()-start, ended=time.time())
            if child is not None:
                row['code'] = child.poll()
            save()
        print(json.dumps(dict(name=name, code=row['code'], seconds=row['seconds'])), flush=True)
        return folder, row

    try:
        folder, _ = worker('sdk-version', [DOTNET, '--version'], build=True)
        assert (folder/'stdout.txt').read_text().strip().endswith('10.0.204')
        flags = ['--tl:off', '--nologo', '-v', 'minimal', '-p:EnableSourceControlManagerQueries=false',
                 '-p:EnableSourceLink=false', '-p:UseSharedCompilation=false', '-nr:false']
        projects = build_prerequisites(base, worker, flags)
        built = {p.relative_to(base).as_posix(): pin(p) for root in
                 [project.parent/'bin' for _, project in projects] for p in root.rglob('*') if p.is_file()}
        write(campaign/'built-files.json', built)
        worker('il-bridge', [DOTNET, base/'il-bridge/bin/Release/net10.0/IlBridge.dll', base/'runtimes/portable',
                             base/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0', campaign/'il-bridge.json'])
        bridge = read(campaign/'il-bridge.json'); assert bridge['passed'] is True
        assert [(r['assembly'], r['methods'], r['equal']) for r in bridge['observations']] == [
            ('Lokad.Onnx.dll', 3113, True), ('Lokad.Onnx.Data.dll', 697, True)]
        suites = {}
        for name, project in projects[:2]:
            worker(name+'-tests', [DOTNET, 'test', project, '-c', 'Release', *flags, '--no-build', '--no-restore',
                                 '--logger', 'trx;LogFileName='+name+'.trx', '--results-directory', campaign/'test-results'], build=True)
            suites[name] = test_results(campaign/'test-results'/(name+'.trx'), 3313 if name == 'backend' else 343,
                                        REQUIRED_TESTS if name == 'backend' else ())
        write(campaign/'operator-gate.json', dict(passed=True, suites=suites, il=pin(campaign/'il-bridge.json')))
        verify(base)
        from fresh_qualification import qualification
        for mode in ('normal', 'disabled'):
            prefix = [] if mode == 'normal' else ['/usr/bin/env', 'DOTNET_EnableHWIntrinsic=0']
            worker('caller-' + mode, [*prefix, DOTNET, base/'caller/Caller.dll', base/'runtimes/production',
                   base/'caller/shapes.json', campaign/('caller-' + mode + '.json'), mode,
                   pin(base/'runtimes/portable/Lokad.Onnx.dll')['sha256']])
        for role in ROLES:
            manifest = base/'manifests'/(role+'-pyannote.json')
            worker(role+'-pyannote', [DOTNET, base/'runtimes'/role/'GraphQualification.dll',
                   base/'assets', manifest, campaign/(role+'-pyannote-output'),
                   pin(base/'runtimes'/role/'Lokad.Onnx.dll')['sha256']])
            worker(role+'-parakeet', [DOTNET, base/'runtimes'/role/'TranscribeReplay.dll',
                   '/home/vermorel/Onnx/models/parakeet-tdt-0.6b-v3',
                   base/'parakeet-reference/manifest.json', campaign/(role+'-parakeet.json')])
        reports = qualification(base, campaign)
        gate(reports)
        native_manifest = base/'manifests/production-pyannote.json'
        native_output = campaign/'native-conformance-output'
        worker('native-conformance', [sys.executable, '-B', base/'runtime/native.py', base/'assets', native_manifest, native_output, 'conformance'])
        native_result(base, native_output, native_manifest, 'conformance', spec)
        verify(base)
        write(campaign/'qualification-gate.json', dict(passed=True, reports=reports, operator_gate=pin(campaign/'operator-gate.json'),
                                                      native=pin(native_output/'result.json')))
        for mode in ('inputs', 'run'):
            worker('meetings-' + mode, [DOTNET, base/'runtimes/portable/NaturalMeetings.dll',
                '/home/vermorel/Onnx', base/'meetings', campaign/('meetings-' + mode + '-output'), mode])
        from meetings_audit import audit_meetings
        meeting_report = audit_meetings(base, campaign)
        write(campaign/'meetings-audit.json', meeting_report)
        assert meeting_report['passed']
        for index, role in enumerate(TIMING_ROLES):
            manifest = native_manifest if role == 'ort' else base/'manifests'/(role+'-pyannote.json')
            output = campaign/f'timing-{index:02}-{role}-output'
            prefix = [sys.executable, '-B', base/'runtime/native.py'] if role == 'ort' else [DOTNET, base/'runtimes'/role/'AudioBenchmark.dll']
            worker(f'timing-{index:02}-{role}', [*prefix, base/'assets', manifest, output, 'timing'])
            if role == 'ort':
                native_result(base, output, manifest, 'timing', spec)
            else:
                result = read(output/'result.json'); wanted = read(manifest)
                validate_records(result, wanted, 'timing')
                assert result['manifest_sha256'] == pin(manifest)['sha256']
                assert result['engine'] == 'managed' and result['runtime'] == '.NET 10.0.8' and result['processor_count'] == 1
                assert result['runner_sha256'] == pin(base/'runtimes'/role/'AudioBenchmark.dll')['sha256']
                for key in ('core_sha256', 'data_sha256'):
                    assert result[key] == wanted[key]
        from candidate_protocol import verified_files
        verified_files(base, built)
        verify(base); state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); traceback.print_exc()
    finally:
        state.update(complete=True, ended=time.time(), seconds=time.monotonic()-started); save()
    return state['code']


if __name__ == '__main__':
    raise SystemExit(run(Path(sys.argv[1]).resolve()))
