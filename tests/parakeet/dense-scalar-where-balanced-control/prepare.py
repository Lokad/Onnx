"""Pin the rejected control and reuse its exact consumer, products and fixtures."""
import ast
import importlib.util
import json
from pathlib import Path
import shutil
import tarfile
from protocol import pin, read, save, DIAGNOSTIC_FLAGS
from fixtures import screen_cases

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-dense-scalar-where-balanced-control-amd-20260924'
QUALIFIED = ROOT / 'artifacts/parakeet-dense-scalar-where-balanced-codegen-amd-20260924'
NATIVE = ROOT / 'tests/parakeet/dense-scalar-where-results/balanced-codegen-review-20260924.json'
CONTROL = ROOT / 'artifacts/parakeet-dense-scalar-where-stability-amd-20260924'
OLD = ROOT / 'tests/parakeet/dense-scalar-where-stability-amd'
CONSUMER = {'bytes': 59392, 'sha256': 'bfd850bcde3d38a0dd29448462780f74312c642218bb90ee632a8538270e0c51'}


def previous_closed():
    assert pin(QUALIFIED/'closed.json')['sha256']=='ef63ec76aaf45d9df155de015b660b606e6e106446b5e6916c2cd3c048942457'
    proof=read(QUALIFIED/'closed.json');assert proof['passed'] and not proof['performance_admitted']
    for name,wanted in proof['files'].items():assert pin(QUALIFIED/name)==wanted,name
    assert pin(NATIVE)['sha256']=='2a6d0ed0beb315c0076d1df1e595646d83b97f4f69400939ec550c02854d623e'
    native=read(NATIVE);assert native['native_admitted'] and native['closure']==pin(QUALIFIED/'closed.json')
    for body in native['bodies']:assert pin(ROOT/body['file'])==body['identity']
    path=ROOT/'tests/parakeet/dense-scalar-where-balanced-codegen/prepare.py'
    spec=importlib.util.spec_from_file_location('qualified_prepare',path)
    previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous);previous.previous_closed()
    for name,wanted in read(QUALIFIED/'prepared.json')['files'].items():assert pin(ROOT/name)==wanted,name
    assert read(QUALIFIED/'collected/built.json')['consumer']==CONSUMER
    for name in ['score.py','test_score.py']:
        expected=(OLD/name).read_text().replace('parakeet-dense-where-whole-census-600-180-v1','parakeet-dense-where-balanced-600-180-v1')
        assert (TOOLS/name).read_text()==expected,name
    for name in ['fixtures.py','original_fixtures.py','numerical_fixtures.py']:
        assert (TOOLS/name).read_bytes()==(OLD/name).read_bytes(),name
    assert (TOOLS/'checks.py').read_bytes()==(ROOT/'tests/parakeet/dense-scalar-where-balanced-codegen/checks.py').read_bytes()


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle=BASE/'bundle'; bundle.mkdir(); originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for folder in ['source','evidence','fixtures']:
        for p in sorted((QUALIFIED/'bundle'/folder).rglob('*')):
            if p.is_file(): copy(p,bundle/p.relative_to(QUALIFIED/'bundle'))
    copy(CONTROL/'bundle/cases.json',bundle/'cases.json')
    copy(QUALIFIED/'collected/built.json',bundle/'built.json')
    products=read(QUALIFIED/'payload.json')['products']
    for role in ['current','candidate']:
        for p in sorted((QUALIFIED/'collected/runtimes'/role).iterdir()):
            if p.is_file(): copy(p,bundle/'runtimes'/role/p.name)
        assert pin(bundle/'runtimes'/role/'ParakeetProviderWhereScreen.dll')==CONSUMER
        for name,wanted in products[role].items(): assert pin(bundle/'runtimes'/role/name)==wanted
    for name in ['closed.json','analysis.json','payload.json']:
        copy(QUALIFIED/name,bundle/'evidence'/('control-'+name))
    copy(QUALIFIED/'collected/collection.json',bundle/'evidence/control-collection.json')
    copy(NATIVE,bundle/'evidence/balanced-codegen-review.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']: copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m61-parakeet-where-balanced-control-20260924.md',bundle/'prospective-plan.md')
    links=read(CONTROL/'bundle/stage.json')['fixture_links']; assert len(links)==32
    save(bundle/'stage.json',dict(passed=True,products=products,consumer=CONSUMER,root_product_changed=False,
        diagnostic_flags=DIAGNOSTIC_FLAGS,fixture_links=links,cases=220,control_processes=4,
        files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
    for p in TOOLS.iterdir():
        if p.is_file():
            if p.suffix=='.py': ast.parse(p.read_text(),str(p))
            originals[p.relative_to(ROOT).as_posix()]=pin(p)
    parser=ROOT/'tests/parakeet/wide-entry-first-use-results/inspect_codegen.py'
    originals[parser.relative_to(ROOT).as_posix()]=pin(parser)
    with tarfile.open(BASE/'payload.tar.gz','w:gz') as archive:
        for p in sorted(bundle.rglob('*')):
            if p.is_file() and not p.relative_to(bundle).as_posix().startswith('fixtures/'):
                archive.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
    save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
    print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),cases=220,consumer=CONSUMER)))


if __name__=='__main__': prepare()
