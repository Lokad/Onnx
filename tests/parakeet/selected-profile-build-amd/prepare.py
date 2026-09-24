"""Freeze the selected release and an exact two-literal diagnostic adaptation."""
import ast
import difflib
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from checks import OLD, CURRENT
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/parakeet-selected-profile-build-amd-20260924'
PRIOR = ROOT/'artifacts/parakeet-current-profile-build-amd-20260923'
APP = ROOT/'artifacts/parakeet-wide-entry-first-use-app-amd-v2-20260923'
ROOT_BUILD = ROOT/'artifacts/parakeet-wide-entry-first-use-root-amd-v2-20260923'
CONTROL = ROOT/'artifacts/parakeet-dense-scalar-where-balanced-control-amd-20260924'
SOURCE = ROOT/'artifacts/parakeet-wide-entry-first-use-source-v2-20260923/prepared.json'
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
spec = importlib.util.spec_from_file_location('selected_profile_monitor', MONITOR)
monitor = importlib.util.module_from_spec(spec); spec.loader.exec_module(monitor)
PROOFS = {
    'prior': (PRIOR, '27df0af0d6ef488a4468aa7cee8f1d31112eebd28525b79ea8d6e6da4fddcecc'),
    'app': (APP, 'ed90f6bab75fc1b6aae2b322fa2d2dec36458ba83364f62ea7588b1df841c0de'),
    'root': (ROOT_BUILD, '16d570819ab69915fe34fa6c5a4efb79d45ae792dbd5b1448c65645fa0d55f73'),
    'control': (CONTROL, 'e9c296b8cc4c3be86a8e8447225e5254fae2f264741da8dcc128ade8d6d6db9c')}


def previous_closed():
    for folder, digest in PROOFS.values():
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder/name) == wanted, name
    assert not read(CONTROL/'closed.json')['stability_admitted']
    assert read(PRIOR/'consumer-review.json')['passed']
    assert pin(PRIOR/'consumer-review.json') == pin(ROOT/'tests/parakeet/current-profile-results/consumer-review-20260923.json')
    source = read(SOURCE); assert source['passed'] and len(source['source']) == 422
    for name,wanted in source['source'].items(): assert pin(ROOT/name) == wanted,name


def prepare():
    assert not BASE.exists(); previous_closed()
    BASE.mkdir(); bundle = BASE/'bundle'; bundle.mkdir(); originals = {}
    def copy(source, target):
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['Program.cs','Diagnostic.cs','NpySupport.cs','SampledAudio.csproj']:
        copy(PRIOR/'bundle/source'/name, bundle/'source'/name)
    copy(ROOT/'global.json', bundle/'source/global.json')
    for source in (APP/'collected/runtimes/candidate').iterdir():
        if source.is_file(): copy(source, bundle/'runtime'/source.name)
    for source in (PRIOR/'collected/runtime').iterdir():
        if source.is_file(): copy(source, bundle/'previous'/source.name)
    product = {name:pin(bundle/'runtime'/name) for name in CURRENT}
    assert {name:wanted['sha256'] for name,wanted in product.items()} == CURRENT
    assert not list((bundle/'runtime').glob('SampledAudio.*'))
    path = bundle/'source/Program.cs'; before = path.read_text(encoding='utf8'); after = before
    for name, old in OLD.items():
        assert after.count(old) == 1 and CURRENT[name] not in after
        after = after.replace(old, CURRENT[name])
    path.write_text(after, encoding='utf8', newline='\n')
    patch = ''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='qualified/Program.cs',tofile='selected/Program.cs'))
    (bundle/'consumer.patch').write_text(patch,encoding='utf8')
    copy(TOOLS/'Bridge.cs.txt', bundle/'bridge-source/Program.cs')
    copy(TOOLS/'Bridge.csproj', bundle/'bridge-source/Bridge.csproj')
    copy(ROOT/'global.json', bundle/'bridge-source/global.json')
    for p in TOOLS.glob('*.py'):
        ast.parse(p.read_text(),str(p))
        if p.name in ['protocol.py','remote.py','remote_prepare.py','checks.py']: copy(p,bundle/'tools'/p.name)
    for label,(folder,_) in PROOFS.items():
        copy(folder/'closed.json', bundle/'evidence'/(label+'-closed.json'))
        copy(folder/'collected/collection.json', bundle/'evidence'/(label+'-collection.json'))
        copy(folder/'payload.json', bundle/'evidence'/(label+'-payload.json'))
    copy(PRIOR/'consumer-review.json',bundle/'evidence/prior-consumer-review.json')
    copy(SOURCE,bundle/'evidence/selected-source.json')
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m62-parakeet-selected-profile-20260924.md',bundle/'prospective-plan.md')
    stage = dict(passed=True,product=product,previous_consumer=pin(bundle/'previous/SampledAudio.dll'),
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    for p in [*TOOLS.iterdir(),MONITOR,SOURCE]:
        if p.is_file(): originals[p.relative_to(ROOT).as_posix()] = pin(p)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),product=product)))


if __name__ == '__main__': prepare()
