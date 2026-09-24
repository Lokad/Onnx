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
BASE = ROOT / 'artifacts/parakeet-dense-scalar-where-control-codegen-amd-20260924'
CONTROL = ROOT / 'artifacts/parakeet-dense-scalar-where-stability-amd-20260924'
OLD = ROOT / 'tests/parakeet/dense-scalar-where-stability-amd'
CONSUMER = dict(bytes=52736, sha256='19f25657ce5abf2af66d6d2e499a8ead0a002c49b3ba5f5c76a03b61ac014dd2')


def previous_closed():
    assert pin(CONTROL/'closed.json')['sha256']=='882ac1267e5ef80fc7b8bf2b01f3252f5b72cba8fb2fc1da2c783fa9e7e27986'
    closure=read(CONTROL/'closed.json'); assert closure['passed'] and not closure['stability_admitted']
    for name,wanted in closure['files'].items(): assert pin(CONTROL/name)==wanted,name
    for name,wanted in read(CONTROL/'prepared.json')['files'].items(): assert pin(ROOT/name)==wanted,name
    spec=importlib.util.spec_from_file_location('control_prepare', OLD/'prepare.py')
    prior=importlib.util.module_from_spec(spec); spec.loader.exec_module(prior); prior.previous_closed()
    assert read(CONTROL/'collected/built.json')['consumer']==CONSUMER
    products=read(CONTROL/'payload.json')['products']
    assert products['current']==products['candidate']==prior.build_products()
    assert all((TOOLS/n).read_bytes()==(OLD/n).read_bytes() for n in ['fixtures.py','original_fixtures.py','numerical_fixtures.py'])
    cases=screen_cases(read(CONTROL/'bundle/evidence/capture-result.json'),read(CONTROL/'bundle/evidence/qualified-reference.json'))
    assert len(cases)==220 and cases==read(CONTROL/'bundle/cases.json')


def prepare():
    assert not BASE.exists(); previous_closed(); BASE.mkdir(); bundle=BASE/'bundle'; bundle.mkdir(); originals={}
    def copy(source,target):
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        originals[source.relative_to(ROOT).as_posix()]=pin(source)
    for folder in ['source','evidence','fixtures']:
        for p in sorted((CONTROL/'bundle'/folder).rglob('*')):
            if p.is_file(): copy(p,bundle/p.relative_to(CONTROL/'bundle'))
    copy(CONTROL/'bundle/cases.json',bundle/'cases.json')
    copy(CONTROL/'collected/built.json',bundle/'built.json')
    products=read(CONTROL/'payload.json')['products']
    for role in ['current','candidate']:
        for p in sorted((CONTROL/'collected/runtimes'/role).iterdir()):
            if p.is_file(): copy(p,bundle/'runtimes'/role/p.name)
        assert pin(bundle/'runtimes'/role/'ParakeetProviderWhereScreen.dll')==CONSUMER
        for name,wanted in products[role].items(): assert pin(bundle/'runtimes'/role/name)==wanted
    for name in ['closed.json','analysis.json','payload.json']:
        copy(CONTROL/name,bundle/'evidence'/('control-'+name))
    copy(CONTROL/'collected/collection.json',bundle/'evidence/control-collection.json')
    for name in ['protocol.py','remote.py','remote_prepare.py']: copy(TOOLS/name,bundle/'tools'/name)
    copy(TOOLS/'README.md',bundle/'README.md')
    shutil.copy2(ROOT/'.agent/m60-parakeet-where-control-diagnostic-20260924.md',bundle/'prospective-plan.md')
    links=read(CONTROL/'bundle/stage.json')['fixture_links']; assert len(links)==32
    save(bundle/'stage.json',dict(passed=True,products=products,consumer=CONSUMER,root_product_changed=False,
        diagnostic_flags=DIAGNOSTIC_FLAGS,fixture_links=links,cases=220,diagnostic_processes=4,
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
