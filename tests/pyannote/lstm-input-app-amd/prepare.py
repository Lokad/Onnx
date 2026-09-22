"""Freeze complete application admission after all product/model gates close."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save

ROOT = Path(__file__).resolve().parents[3]; TOOLS = Path(__file__).resolve().parent
BASE = ROOT/'artifacts/pyannote-lstm-input-app-amd-20260922'
APP = ROOT/'artifacts/pyannote-blocked-spatial-app-amd-execution-20260922'
APP_PAYLOAD = ROOT/'artifacts/pyannote-blocked-spatial-app-amd-payload-20260922/payload'
PRIOR = {
    'product': ('product-amd-v2','7d442059d85e5df31f9a616b4e5e5d5203a612affb41b59a0c1128e6e046529f'),
    'models': ('models-amd','2a348f67238a44e3cc36f58e8cd5bf467fa1a8562c57c788a04902f8bdb8c90b'),
    'parakeet': ('parakeet-amd','36c475cfde426125dc58df4f40f14318d5e55fb8bc6c9b68b225c7885706d17f'),
    'shared': ('shared-amd','484050a96b08c42debd52b1cca77674969f6a09cea192894cb9142cd13e150ae')}
MONITOR = ROOT/'tests/parakeet/packing-budgets/common.py'
module = importlib.util.spec_from_file_location('application_monitor', MONITOR)
monitor = importlib.util.module_from_spec(module); module.loader.exec_module(monitor)


def previous_closed():
    for suffix,digest in [*PRIOR.values()]:
        folder = ROOT/'artifacts'/('pyannote-lstm-input-'+suffix+'-20260922')
        assert pin(folder/'closed.json')['sha256'] == digest
        proof = read(folder/'closed.json'); assert proof['passed']
        for name,wanted in proof['files'].items(): assert pin(folder/name) == wanted,name
    assert pin(APP/'closed.json')['sha256'] == '5c238cd33845eb58fc00332361a967185ae82a9fcb70854530e07fad58f064d0'
    for name,wanted in read(APP/'closed.json')['files'].items(): assert pin(APP/name) == wanted,name
    assert pin(APP_PAYLOAD/'payload.json')['sha256'] == '229556d67d87085875ade0fc027a1b1df327f5578c5c60d27ad961462aac534f'


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir()
    bundle = BASE/'bundle'; bundle.mkdir(); originals = {}; prerequisites = {}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()] = pin(source)
    for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','meeting_protocol.py','meetings_audit.py','admission.py']:
        copy(TOOLS/name,bundle/'tools'/name)
    for name,(suffix,digest) in PRIOR.items():
        folder = ROOT/'artifacts'/('pyannote-lstm-input-'+suffix+'-20260922')
        for filename in ['closed.json','analysis.json']: copy(folder/filename,bundle/'evidence'/name/filename)
        prerequisites[name] = dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))
    copy(APP/'execution/execution.json',bundle/'evidence/original-execution.json')
    for family in ['pyannote','parakeet']:
        source = APP_PAYLOAD/'manifests'/('portable-'+family+'.json')
        assert pin(source) == read(APP_PAYLOAD/'payload.json')['files'][source.relative_to(APP_PAYLOAD).as_posix()]
        copy(source,bundle/'evidence'/('original-'+family+'.json'))
    copy(APP_PAYLOAD/'meetings/manifest.json',bundle/'evidence/original-meetings.json')
    copy(ROOT/'PLAN.md',bundle/'prospective-plan.md'); originals.pop('PLAN.md')
    identities = read(bundle/'evidence/models/analysis.json')['identities']
    original = read(APP_PAYLOAD/'payload.json')
    stage = dict(passed=True,identities=identities,prerequisites=prerequisites,
        native_inventory=pin(APP/'execution/execution.json'),
        consumers={name:original['files']['runtimes/portable/'+name+'.dll'] for name in ['AudioBenchmark','NaturalMeetings']},
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
    save(bundle/'stage.json',stage)
    files = dict(originals)
    for p in [*TOOLS.iterdir(),MONITOR,APP/'closed.json',APP_PAYLOAD/'payload.json']:
        if p.is_file(): files[p.relative_to(ROOT).as_posix()] = pin(p)
    for p in TOOLS.glob('*.py'): ast.parse(p.read_text(),str(p))
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
        for p in sorted(bundle.rglob('*')):
            if p.is_file(): tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=files,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'))))


if __name__ == '__main__': prepare()
