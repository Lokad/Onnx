"""Retain full graph/native/public checks and prove the consumer literal edit."""
import shutil
import sys
import traceback
from common import *

OLD = ROOT / 'artifacts/pyannote-combined-consumers-20260922'
OLD_DATA = '85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'
PRIOR = ROOT / 'artifacts/parakeet-portable-models-20260922/pyannote'
QUALIFIER = ROOT / 'tests/parakeet/reduction-shared/qualify_pyannote.py'


def prepare():
    candidate()
    assert not BASE.exists()
    assert pin(OLD / 'closed.json')['sha256'] == '602d752956d6cf830851cd0dc35eb4bdf1230880fb2dce6d8bbea6f530bb555c'
    previous = read(OLD / 'closed.json')
    assert previous['passed']; verify(previous['files'])
    for identity in previous['identities']: terminal(identity)
    # This prior full-model composition independently established unchanged
    # Pyannote output bytes relative to the selected portable implementation.
    comparison = read(PRIOR / 'analysis.json')
    assert comparison['passed'] and not any(r['changed_bits'] for r in comparison['arrays'])
    prior = read(PRIOR / 'closed.json'); assert prior['passed']; verify(prior['files'])
    for identity in prior['terminal_identities']: terminal(identity)
    BASE.mkdir(); (BASE / 'logs').mkdir()
    runtime = BASE / 'runtime'; shutil.copytree(MODEL / 'runtime', runtime)
    consumer = BASE / 'consumer'; consumer.mkdir()
    for name in ['Program.cs', 'NpySupport.cs', 'GraphQualification.csproj']:
        shutil.copy2(OLD / 'portable' / name, consumer / name)
    p = consumer / 'Program.cs'; before = p.read_text(encoding='utf8')
    assert before.count(OLD_DATA) == 1
    p.write_text(before.replace(OLD_DATA, DATA), encoding='utf8')
    assert p.read_text(encoding='utf8').replace(DATA, OLD_DATA) == before
    files = {}
    def bind(path): files[rel(path)] = pin(path)
    for folder in [runtime, consumer, TOOLS, OLD / 'bridge/bin/Release/net10.0']:
        for p in folder.iterdir():
            if p.is_file(): bind(p)
    for p in [MODEL / 'closed.json', OLD / 'closed.json', PRIOR / 'closed.json', PRIOR / 'analysis.json',
              PRIOR / 'output/result.json', QUALIFIER, MONITOR,
              ROOT / 'tests/parakeet/portable-models/common.py', INPUT,
              ROOT / 'artifacts/pyannote-performance-profile-20260921/output/result.json']:
        bind(p)
    manifest = read(INPUT)
    specs = list(manifest['models'].values()) + [manifest['reference']] + [c['pcm'] for c in manifest['cases']]
    specs += list(manifest.get('native_assets', {}).values())
    upstream = manifest['upstream']; specs += [upstream] if 'path' in upstream else list(upstream.values())
    for item in specs:
        p = ROOT / item['path']; assert pin(p) == {k: item[k] for k in ['bytes', 'sha256']}; bind(p)
    reference = ROOT / manifest['reference']['path']
    for name, item in read(reference)['files'].items():
        p = reference.parent / name; assert pin(p) == {k: item[k] for k in ['bytes', 'sha256']}; bind(p)
    save(BASE / 'inputs.json', dict(passed=True, files=files))
    monitor.BASE = BASE; own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    flags = monitor.FLAGS + ['-p:NuGetAudit=false']; project = consumer / 'GraphQualification.csproj'
    extra = ['-p:FrozenProductDirectory=' + str(runtime)]
    try:
        jobs = [('consumer-restore', ['dotnet', 'restore', project, *flags, '--source', FEED, '--packages', BASE / 'packages', *extra]),
                ('consumer-build', ['dotnet', 'build', project, '-c', 'Release', *flags, '--no-restore', '--disable-build-servers', *extra]),
                ('consumer-instructions', ['dotnet', OLD / 'bridge/bin/Release/net10.0/Bridge.dll',
                  OLD / 'portable/bin/Release/net10.0', consumer / 'bin/Release/net10.0', BASE / 'consumer-instructions.json'])]
        for name, args in jobs:
            monitor.worker(state, BASE / 'build-state.json', name, args, ROOT, [0], 8, 8, 900, True, consumer)
            print(name, 'passed', flush=True)
        inventory = read(BASE / 'consumer-instructions.json')
        assert inventory['inventory_complete'] and len(inventory['observations']) == 1
        row = inventory['observations'][0]
        assert row['methods'] == 96 and row['public_surface_equal'] and not row['added'] and not row['removed']
        assert len(row['differences']) == 1
        method = row['differences'][0]; assert method.startswith('Program::<Main>$::')
        old, new = row['normalized_methods'][method], row['candidate_methods'][method]
        assert old.count(OLD_DATA) == 1 and old.replace(OLD_DATA, DATA) == new
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            shutil.copy2(consumer / 'bin/Release/net10.0' / ('GraphQualification.' + suffix), runtime / ('GraphQualification.' + suffix))
        verify(files)
        for p in runtime.iterdir():
            if p.is_file(): bind(p)
        save(BASE / 'prepared.json', dict(passed=True, files=files, core=CORE, data=DATA,
             consumer_methods=96, unchanged_consumer_methods=95, changed_literal=dict(before=OLD_DATA, after=DATA)))
        state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'build-state.json', state)


def run():
    spec = read(BASE / 'prepared.json'); assert spec['passed']; verify(spec['files'])
    resources(BASE, 'build-state.json', {n: (8, 8, 900, False) for n in ['consumer-restore', 'consumer-build', 'consumer-instructions']})
    assert not (BASE / 'processes.json').exists()
    monitor.BASE = BASE; own = psutil.Process()
    state = dict(complete=False, code=None, supervisor=dict(pid=own.pid, birth=own.create_time()), runs=[])
    try:
        monitor.worker(state, BASE / 'processes.json', 'pyannote',
                       ['dotnet', BASE / 'runtime/GraphQualification.dll', ROOT, INPUT, BASE / 'output', CORE],
                       ROOT, [0], 10, 8, 900, False, BASE / 'output')
        verify(spec['files']); state['code'] = 0
    except BaseException:
        state.update(code=1, error=traceback.format_exc()); raise
    finally:
        state['complete'] = True; save(BASE / 'processes.json', state)


def audit():
    built = resources(BASE, 'build-state.json', {n: (8, 8, 900, False) for n in ['consumer-restore', 'consumer-build', 'consumer-instructions']})
    executed = resources(BASE, 'processes.json', {'pyannote': (10, 8, 900, True)})
    current, previous = read(BASE / 'output/result.json'), read(PRIOR / 'output/result.json')
    assert current['core_sha256'] == CORE and current['data_sha256'] == DATA
    assert len(current['rows']) == 18 and len(current['applications']) == 16
    for row, old in zip(current['rows'], previous['rows'], strict=True):
        for kind in ['input', 'output']: assert row[kind] == old[kind]
    for row, old in zip(current['applications'], previous['applications'], strict=True):
        assert (row['name'], row['pass'], row['result']) == (old['name'], old['pass'], old['result'])
    # Every original native, numerical, repeat, ownership and public assertion.
    source = QUALIFIER.read_text(encoding='utf8')
    old = "sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reduction-dispatch'))\nfrom common import ROOT, BASE as MODEL, pin, read, save, verify, terminal, psutil, monitor, rel"
    assert source.count(old) == 1
    source = source.replace(old, 'from common import ROOT, MODEL, pin, read, save, verify, terminal, psutil, monitor, rel')
    namespace = dict(__name__='direct_original_graph_auditor', __file__=str(QUALIFIER))
    exec(compile(source, str(QUALIFIER), 'exec'), namespace)
    namespace.update(BASE=BASE, MODEL=MODEL, CORE=CORE)
    namespace['audit']()
    identities = built['identities'] + executed['identities']
    result = dict(passed=True, exact_predecessor_arrays=18, exact_predecessor_public_requests=16,
                  consumer_methods=96, unchanged_consumer_methods=95, identities=identities,
                  resources=built['resources'] + executed['resources'], native_closure=pin(BASE / 'closed.json'))
    assert not (BASE / 'complete.json').exists()
    save(BASE / 'complete.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['prepare', 'run', 'audit']
    globals()[sys.argv[1]]()
