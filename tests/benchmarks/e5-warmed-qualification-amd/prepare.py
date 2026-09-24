"""Freeze one uninstrumented case after the complete runtime diagnostic."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
from scope import verify as scope

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/e5-warmed-qualification-amd-20260924'
GRAPH=ROOT/'artifacts/parakeet-validated-composition-graphs-amd-20260924'
DIAGNOSTIC=ROOT/'artifacts/e5-runtime-diagnostic-amd-20260924'
REMOTE_GRAPH='/dev/shm/lokad-parakeet-validated-composition-graphs-20260924'
REMOTE_DIAGNOSTIC='/dev/shm/lokad-e5-runtime-diagnostic-20260924'
CONSUMER='d827e3b9f1e5158e10bd24a5ca009fa7950fd08d08bb5260b84e36b5a853ac02'


def previous_closed():
    scope()
    for folder,digest in [(GRAPH,'729814e9effff9c5e1456e165e7ef792bf2cba1c4978ce28ee3cb3ea7b4d1209'),
                          (DIAGNOSTIC,'8304dc71470c5666b83078f1fa487ae8595259d3ce9738471f70065d4919e990')]:
        assert pin(folder/'closed.json')['sha256']==digest
        closure=read(folder/'closed.json');assert closure['passed']
        for name,wanted in closure['files'].items():assert pin(folder/name)==wanted,name
    assert not read(GRAPH/'closed.json')['admitted']
    assert read(DIAGNOSTIC/'closed.json')['diagnostic_only']
    composition=read(ROOT/'artifacts/parakeet-validated-composition-source-v2-20260924/prepared.json')
    assert len(composition['before'])==422
    for name,wanted in composition['before'].items():assert pin(ROOT/name)==wanted,name
    source=ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs'
    assert pin(source)==read(ROOT/'artifacts/warmed-release-amd-v2-20260923/prepared.json')['files'][source.relative_to(ROOT).as_posix()]
    assert pin(GRAPH/'collected/runtimes/current/ReleaseBenchmark.dll')['sha256']==CONSUMER


def prepare():
    previous_closed();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals=scope()
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','native.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'Program.cs',bundle/'source/consumer/Program.cs')
    for name in ['NpySupport.cs','ReleaseBenchmark.csproj']:
        copy(ROOT/'tests/benchmarks/release-amd-v2'/name,bundle/'source/consumer'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m69-e5-warmup-qualification-20260924.md',bundle/'prospective-plan.md')
    for label,folder in [('graph',GRAPH),('diagnostic',DIAGNOSTIC)]:
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    graph=read(GRAPH/'payload.json');diagnostic=read(DIAGNOSTIC/'payload.json')
    assert graph['products']==diagnostic['products']
    cases=read(GRAPH/'collected/cases.json');cases['cases']=[c for c in cases['cases'] if c['key']=='e5-30tok']
    assert len(cases['cases'])==1;save(bundle/'cases.json',cases)
    links={};original_files=read(GRAPH/'collected/collection.json')['files']
    for name,wanted in original_files.items():
        if name.startswith('runtimes/current/'):
            links[name.replace('runtimes/current/','previous/')]=dict(source=REMOTE_GRAPH+'/'+name,identity=wanted)
        if name.startswith(('runtimes/current/','runtimes/candidate/')) and not Path(name).name.startswith('ReleaseBenchmark.'):
            links[name]=dict(source=REMOTE_GRAPH+'/'+name,identity=wanted)
    for name in ['Lokad.Onnx.dll','Google.Protobuf.dll']:
        key='runtimes/current/'+name;links['product/'+name]=dict(source=REMOTE_GRAPH+'/'+key,identity=original_files[key])
    for row in [*cases['cases'][0]['inputs'],*cases['cases'][0]['outputs']]:
        if 'file' in row:
            name=row['file'];links[name]=dict(source=REMOTE_GRAPH+'/'+name,identity=graph['files'][name])
    for name,wanted in read(DIAGNOSTIC/'collected/collection.json')['files'].items():
        if name.startswith('bridge/'):
            links[name]=dict(source=REMOTE_DIAGNOSTIC+'/'+name,identity=wanted)
    # Preserve the full original runtime environment; omit only the other model assets.
    external={name:wanted for name,wanted in graph['external'].items()
        if not name.startswith('/home/vermorel/Onnx/models/') or name==cases['cases'][0]['model']}
    for name,wanted in diagnostic['external'].items():assert external.setdefault(name,wanted)==wanted,name
    save(bundle/'stage.json',dict(passed=True,links=links,products=graph['products'],
        previous_consumer=pin(GRAPH/'collected/runtimes/current/ReleaseBenchmark.dll'),
        feed=diagnostic['feed'],external=external,interpreter=graph['interpreter'],python_paths=graph['python_paths'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),links=len(links))))


if __name__=='__main__':prepare()
