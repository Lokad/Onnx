"""Freeze four instrumented e5 processes from a failed, fully retained comparison."""
import ast, json, shutil, tarfile
from pathlib import Path
from protocol import pin, read, save
from source_scope import instrument, verify

ROOT = Path(__file__).resolve().parents[3]; TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/e5-runtime-diagnostic-amd-20260924'
GRAPH = ROOT/'artifacts/parakeet-validated-composition-graphs-amd-20260924'
OLD = ROOT/'artifacts/parakeet-dispatch-events-amd-20260923'
OLD_TOOLS = ROOT/'tests/parakeet/dispatch-events-amd'
REMOTE_GRAPH = '/dev/shm/lokad-parakeet-validated-composition-graphs-20260924'
REMOTE_OLD = '/dev/shm/lokad-parakeet-dispatch-events-20260923'


def previous_closed():
    assert pin(GRAPH/'closed.json')['sha256'] == '729814e9effff9c5e1456e165e7ef792bf2cba1c4978ce28ee3cb3ea7b4d1209'
    proof = read(GRAPH/'closed.json'); assert proof['passed'] and not proof['admitted']
    for n, v in proof['files'].items(): assert pin(GRAPH/n) == v, n
    old = read(OLD/'closed.json'); assert old['passed'] and old['diagnostic_only']
    for n, v in old['files'].items(): assert pin(OLD/n) == v, n
    selected = read(ROOT/'artifacts/parakeet-validated-composition-source-v2-20260924/prepared.json')['before']
    assert len(selected) == 422
    for n, v in selected.items(): assert pin(ROOT/n) == v, n
    # Bind the source of the measured warmed consumer, not just a similar file.
    warm = ROOT/'artifacts/warmed-release-amd-v2-20260923'
    warm_proof = read(warm/'closed.json'); assert warm_proof['passed']
    for n,v in warm_proof['files'].items(): assert pin(warm/n)==v,n
    source = ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs'
    assert pin(source) == read(warm/'prepared.json')['files'][source.relative_to(ROOT).as_posix()]


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(p, q):
        q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q); originals[p.relative_to(ROOT).as_posix()] = pin(p)
    for name in ['ClockProbe.cs', 'Producer.csproj']: copy(TOOLS/name, bundle/'source/consumer'/name)
    source = ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs'
    copy(source, bundle/'evidence/OriginalProgram.cs.txt')
    actual = instrument(source.read_text()); (bundle/'source/consumer/Program.cs').write_text(actual, encoding='utf8')
    save(bundle/'evidence/source-review.json', verify(source.read_text(), actual))
    copy(ROOT/'tests/benchmarks/release-amd-v2/NpySupport.cs', bundle/'source/consumer/NpySupport.cs')
    copy(OLD_TOOLS/'ExportAll.cs.txt', bundle/'source/exporter/Export.cs')
    copy(OLD_TOOLS/'Exporter.csproj', bundle/'source/exporter/Exporter.csproj')
    copy(ROOT/'global.json', bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','il_normalization.py']: copy(TOOLS/name, bundle/'tools'/name)
    copy(OLD_TOOLS/'remote.py', bundle/'tools/remote_base.py')
    copy(TOOLS/'README.md', bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m68-e5-runtime-diagnosis-20260924.md', bundle/'prospective-plan.md')
    copy(GRAPH/'closed.json', bundle/'evidence/graph-closed.json')
    copy(GRAPH/'collected/collection.json', bundle/'evidence/graph-collection.json')
    copy(OLD/'collected/collection.json', bundle/'evidence/tracer-collection.json')
    warm = ROOT/'artifacts/warmed-release-amd-v2-20260923'
    copy(warm/'collected/collection.json', bundle/'evidence/bridge-collection.json')
    links = {}; payload = read(GRAPH/'payload.json')
    for role in ['current','candidate']:
        copy(GRAPH/f'collected/timing-e5-30tok-{role}-a/output/result.json', bundle/f'evidence/original-{role}.json')
        manifest = read(GRAPH/f'collected/cases-{role}.json'); case = next(c for c in manifest['cases'] if c['key']=='e5-30tok')
        manifest['cases'] = [case]; save(bundle/f'cases-{role}.json', manifest)
        for n, v in payload['files'].items():
            if n.startswith(f'runtimes/{role}/') and not Path(n).name.startswith('ReleaseBenchmark.'):
                links[n] = dict(source=REMOTE_GRAPH+'/'+n, identity=v)
        for row in [*case['inputs'], *case['outputs']]:
            if 'file' in row:
                n = row['file']; links[n] = dict(source=REMOTE_GRAPH+'/'+n, identity=payload['files'][n])
    old = read(OLD/'payload.json')
    for n, v in old['files'].items():
        if n.startswith('tracer/'): links[n] = dict(source=REMOTE_OLD+'/'+n, identity=v)
    for n,v in read(warm/'collected/collection.json')['files'].items():
        if n.startswith('bridge/'):
            links[n] = dict(source='/dev/shm/lokad-warmed-release-v2-20260923/'+n,identity=v)
    original_files = read(GRAPH/'collected/collection.json')['files']
    for n,v in original_files.items():
        if n.startswith('runtimes/current/'):
            links[n.replace('runtimes/current/','previous/')] = dict(source=REMOTE_GRAPH+'/'+n,identity=v)
    external = dict(old['external']); external[case['model']] = payload['external'][case['model']]
    save(bundle/'stage.json', dict(passed=True, links=links, products=payload['products'], external=external,
        previous_consumer=original_files['runtimes/current/ReleaseBenchmark.dll'],
        feed=old['feed'], interpreter=old['interpreter'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py': ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(passed=True,archive=pin(BASE/'payload.tar.gz'),links=len(links))))


if __name__=='__main__': prepare()
