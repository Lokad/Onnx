"""Freeze a baseline-only diagnostic from retained graph and event evidence."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/graph-startup-diagnostic-amd-20260923'
GRAPH=ROOT/'artifacts/parakeet-first-use-kernels-graphs-amd-20260923'
OLD=ROOT/'artifacts/parakeet-dispatch-events-amd-20260923'
OLD_TOOLS=ROOT/'tests/parakeet/dispatch-events-amd'
REMOTE_GRAPH='/dev/shm/lokad-parakeet-first-use-kernels-graphs-20260923'
REMOTE_OLD='/dev/shm/lokad-parakeet-dispatch-events-20260923'

def previous_closed():
    assert pin(GRAPH/'closed.json')['sha256']=='0ba2059367fdced94d678c98fd6d5fc2ae3d4d5dafd2c8dc537c442c80756a2d'
    proof=read(GRAPH/'closed.json');assert proof['passed'] and not proof['admitted']
    for n,v in proof['files'].items():assert pin(GRAPH/n)==v,n
    selected=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923/bundle/stage.json'
    files={n.removeprefix('source/'):v for n,v in read(selected)['files'].items() if n.startswith('source/')}
    assert len(files)==420
    for n,v in files.items():assert pin(ROOT/n)==v,n

def prepare():
    assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
    def copy(p,q):
        q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);originals[p.relative_to(ROOT).as_posix()]=pin(p)
    for name in ['Program.cs','Producer.csproj']:copy(TOOLS/name,bundle/'source/consumer'/name)
    copy(ROOT/'tests/benchmarks/release-amd-v2/NpySupport.cs',bundle/'source/consumer/NpySupport.cs')
    copy(OLD_TOOLS/'ExportAll.cs.txt',bundle/'source/exporter/Export.cs')
    copy(OLD_TOOLS/'Exporter.csproj',bundle/'source/exporter/Exporter.csproj')
    copy(ROOT/'global.json',bundle/'source/global.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(OLD_TOOLS/'remote.py',bundle/'tools/remote_base.py')
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m44-graph-startup-diagnostic-20260923.md',bundle/'prospective-plan.md')
    copy(GRAPH/'closed.json',bundle/'evidence/graph-closed.json')
    copy(GRAPH/'collected/collection.json',bundle/'evidence/graph-collection.json')
    copy(GRAPH/'collected/timing-gpt2-current-a/output/result.json',bundle/'evidence/original-gpt2.json')
    manifest=read(GRAPH/'collected/cases-current.json');case=next(c for c in manifest['cases'] if c['key']=='gpt2')
    manifest['cases']=[case];save(bundle/'cases.json',manifest)
    links={};payload=read(GRAPH/'payload.json')
    for n,v in payload['files'].items():
        if n.startswith('runtimes/current/'):
            for role in ['a','b']:links[n.replace('runtimes/current/','runtimes/'+role+'/')]=dict(source=REMOTE_GRAPH+'/'+n,identity=v)
    for row in [*case['inputs'],*case['outputs']]:
        if 'file' in row:
            n=row['file'];links[n]=dict(source=REMOTE_GRAPH+'/'+n,identity=payload['files'][n])
    old=read(OLD/'payload.json')
    for n,v in old['files'].items():
        if n.startswith('tracer/'):links[n]=dict(source=REMOTE_OLD+'/'+n,identity=v)
    # Only reuse the existing SDK/feed/runtime dependencies, plus this one model.
    external=dict(old['external']);external[case['model']]=payload['external'][case['model']]
    save(bundle/'stage.json',dict(passed=True,links=links,products=dict(current=payload['products']['current']),
        external=external,feed=old['feed'],interpreter=old['interpreter'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    source=(ROOT/'tests/benchmarks/release-amd-v2/Program.cs').read_text();actual=(TOOLS/'Program.cs').read_text()
    for start,end in [('    graph.Reset();','    long end = Stopwatch.GetTimestamp();'),('    Require(graph.Outputs.Keys','    clocks.Add(')]:
        assert source[source.index(start):source.index(end)] in actual
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(passed=True,archive=pin(BASE/'payload.tar.gz'),links=len(links))))

if __name__=='__main__':prepare()
