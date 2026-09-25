"""Freeze a fresh candidate comparison using the already qualified warmed consumer."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
from checks import consumer_inventory,e5_consumer_inventory
from consumer_scope import verify_scope

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-slice-dense-conversion-graphs-amd-20260925'
OLD=ROOT/'artifacts/release-graph-baseline-amd-v2-20260923'
WARM=ROOT/'artifacts/warmed-release-amd-v2-20260923'
E5=ROOT/'artifacts/e5-warmed-qualification-amd-20260924'
BUILD=ROOT/'artifacts/parakeet-slice-dense-conversion-models-amd-20260925'
APP=ROOT/'artifacts/parakeet-slice-dense-conversion-app-amd-20260925'
SHARED=ROOT/'artifacts/parakeet-slice-dense-conversion-shared-amd-20260925'
PYANNOTE=ROOT/'artifacts/parakeet-slice-dense-conversion-pyannote-amd-20260925'
PRIOR=dict(baseline=OLD,warmed=WARM,e5=E5,build=BUILD,app=APP,shared=SHARED,pyannote=PYANNOTE)
CONSUMER='d827e3b9f1e5158e10bd24a5ca009fa7950fd08d08bb5260b84e36b5a853ac02'


def previous_closed():
    verify_scope()
    assert pin(BUILD/'closed.json')['sha256']=='5860fc366c3d976907f35f8d7a1832c10566fa08910af32c168dca7a63979842'
    pins={OLD:'7bfcfa23a9dcc3aa2b6bff22acf612fd140a82bff62a5d7bf891d92da420be9a',
          WARM:'921e1303a193f0f57b2ccd47fd98f6c378d26c617a10d271e6ab4f94b901b175',
          E5:'71c6d952a4ef99aba8b9953c47c2ed9cd70035691e81f99e3f42761fbb8944a4'}
    for folder,wanted in pins.items():assert pin(folder/'closed.json')['sha256']==wanted
    assert read(OLD/'closed.json')['all_controls_passed'] and read(APP/'closed.json')['admitted']
    for folder in PRIOR.values():
        proof=read(folder/'closed.json');assert proof['passed']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
    spec=read(WARM/'payload.json');built=read(WARM/'collected/built.json')
    review=consumer_inventory(read(WARM/'collected/consumer-inventory/instructions.json'),spec,built)
    assert review==read(WARM/'collected/consumer-inventory/review.json')
    assert built['consumer']==pin(WARM/'collected/runtimes/current/ReleaseBenchmark.dll')
    assert built['consumer']['sha256']==CONSUMER
    selected=read(ROOT/'artifacts/parakeet-slice-dense-conversion-source-20260924/prepared.json')
    assert selected['passed'] and len(selected['source'])==428
    assert len(selected['before'])==427
    for name,wanted in selected['before'].items():assert pin(ROOT/name)==wanted,name
    proof=read(E5/'closed.json');assert proof['passed'] and proof['admitted']
    e5=e5_consumer_inventory(read(E5/'collected/consumer-inventory/instructions.json'),read(E5/'payload.json'),read(E5/'collected/built.json'))
    assert e5==read(E5/'collected/consumer-inventory/review.json')==read(E5/'analysis.json')['consumer']
    assert e5['previous_consumer']==built['consumer']


def prepare():
    previous_closed();assert not BASE.exists();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals=verify_scope()
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','checks_e5.py','native.py','native-e5.py']:copy(TOOLS/name,bundle/'tools'/name)
    copy(ROOT/'global.json',bundle/'source/global.json')
    for label,folder in PRIOR.items():
        for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/label/name)
        copy(folder/'collected/collection.json',bundle/'evidence'/label/'collection.json')
    for name in ['instructions.json','review.json']:
        copy(WARM/'collected/consumer-inventory'/name,bundle/'evidence/warmed-consumer'/name)
    copy(WARM/'collected/built.json',bundle/'evidence/warmed-consumer/built.json')
    for name in ['instructions.json','review.json']:
        copy(E5/'collected/consumer-inventory'/name,bundle/'evidence/e5-consumer'/name)
    copy(E5/'collected/built.json',bundle/'evidence/e5-consumer/built.json')
    copy(OLD/'bundle/cases.json',bundle/'cases.json')
    copy(ROOT/'artifacts/parakeet-slice-dense-conversion-source-20260924/prepared.json',bundle/'evidence/source-prepared.json')
    shutil.copy2(ROOT/'.agent/m73-parakeet-slice-dense-conversion-20260924.md',bundle/'prospective-plan.md')
    copy(TOOLS/'README.md',bundle/'README.md')
    old=read(OLD/'payload.json');build=read(BUILD/'analysis.json');warm=read(WARM/'collected/built.json')
    products={role:{'Lokad.Onnx.dll':build['identities'][label]['Lokad.Onnx.dll']} for role,label in [('current','selected'),('candidate','candidate')]}
    for folder in [SHARED,PYANNOTE]:
        identities=read(folder/'analysis.json')['identities']
        assert identities['selected']['Lokad.Onnx.dll']==products['current']['Lokad.Onnx.dll']
        assert identities['candidate']['Lokad.Onnx.dll']==products['candidate']['Lokad.Onnx.dll']
    app=read(APP/'analysis.json')['identities']
    assert all(app[role]['Lokad.Onnx.dll']==products[role]['Lokad.Onnx.dll'] for role in products)
    consumer_files={n.removeprefix('runtimes/current/'):v for n,v in warm['files'].items() if n.startswith('runtimes/current/')}
    assert set(consumer_files)=={'ReleaseBenchmark.'+s for s in ['dll','deps.json','runtimeconfig.json']}
    for name,wanted in consumer_files.items():
        source=WARM/'collected/runtimes/current'/name;assert pin(source)==wanted
        originals[source.relative_to(ROOT).as_posix()]=wanted
    e5=read(E5/'collected/built.json')
    e5_files={n.removeprefix('runtimes/current/'):v for n,v in e5['files'].items() if n.startswith('runtimes/current/')}
    assert set(e5_files)==set(consumer_files)
    for name,wanted in e5_files.items():
        source=E5/'collected/runtimes/current'/name;assert pin(source)==wanted
        originals[source.relative_to(ROOT).as_posix()]=wanted
    stage=dict(passed=True,products=products,consumer=warm['consumer'],consumer_files=consumer_files,
        e5_consumer=e5['consumer'],e5_consumer_files=e5_files,e5_previous_consumer=warm['consumer'],
        previous_consumer=read(WARM/'payload.json')['previous_consumer'],external=old['external'],
        interpreter=old['interpreter'],python_paths=old['python_paths'],
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py':ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz',dereference=True) as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__=='__main__':prepare()
