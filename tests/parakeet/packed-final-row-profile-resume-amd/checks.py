"""Validate the retained control with the original per-process audit body."""
import ast
import importlib.util
from pathlib import Path
import sys
import textwrap
from rules import verify_refusal

ROOT=Path(__file__).resolve().parents[3]
ORIGINAL=ROOT/'tests/parakeet/packed-final-row-profile-amd'
sys.path.insert(0,str(ORIGINAL))
import run as original_transport
PROFILE=original_transport.BASE
APP=original_transport.APP
pin,read=original_transport.pin,original_transport.read


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def original_checker():
    original=module('retained_phase_auditor',ORIGINAL/'audit.py')
    source=(ORIGINAL/'audit.py').read_text(encoding='utf8');tree=ast.parse(source)
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    loop,=[n for n in main.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='run']
    body=textwrap.dedent('\n'.join(source.splitlines()[loop.body[0].lineno-1:loop.end_lineno]))
    generated='def check_run(run,state,folder,spec,built,expected,protocol,accounting,manifest,results,phases,resources):\n'+textwrap.indent(body,'    ')+'\n'
    parsed=ast.parse(generated).body[0]
    assert [ast.dump(n) for n in parsed.body]==[ast.dump(n) for n in loop.body]
    scope=dict(vars(original));exec(compile(generated,'unchanged-retained-process-audit','exec'),scope)
    return scope['check_run'],original.attribute


def partial():
    original_transport.prepared()
    folder=PROFILE/'capture-collected';receipt=read(folder/'capture-collection.json')
    transfer=read(PROFILE/'capture-transfer.json')
    assert transfer['passed'] and transfer['collection']==pin(folder/'capture-collection.json')
    assert transfer['archive']==pin(PROFILE/'capture-results.tar.gz')
    assert receipt['terminal'] and receipt['code']==1
    for name,wanted in receipt['files'].items():assert pin(folder/name)==wanted,name
    state=read(folder/'capture-state.json');spec=read(PROFILE/'bundle/spec.json')
    assert receipt['state']==pin(folder/'capture-state.json')
    assert state['supervisor']==read(PROFILE/'capture-deployment.json')
    refusal=verify_refusal(state,spec)
    assert not any(name.startswith(('phase/','wall/','logs/phase.','logs/wall.')) for name in receipt['files'])
    built=read(folder/'built.json');review=read(PROFILE/'build-review.json')
    assert review['passed'] and review['built']==pin(folder/'built.json')
    assert review['core_unchanged'] and review['consumer_unchanged'] and review['constructor_unchanged']
    assert built['core']==spec['core'] and built['consumer']==spec['original_consumer']
    correction=read(PROFILE/'build-review-correction.json')
    assert correction['corrected_reviewer']==review['reviewer']==pin(PROFILE/'build-review-corrected.py')
    assert correction['original_reviewer']==pin(ORIGINAL/'review_build.py')==pin(PROFILE/'build-review-original.py')
    before=(ORIGINAL/'review_build.py').read_text(encoding='utf8')
    assert (PROFILE/'build-review-corrected.py').read_text(encoding='utf8')==before.replace(
        "if 'warning' in line.lower()]","if 'warning' in line.lower() and line.strip() != '0 Warning(s)']",1)
    reference=read(PROFILE/'bundle/evidence/candidate-public.json')
    expected={r['name']:r['result'] for r in reference['records']};assert len(expected)==20
    assert reference['core_sha256']==spec['core']['sha256'] and reference['data_sha256']==spec['data']['sha256']
    protocol=module('retained_protocol',APP/'collected/runtime/protocol.py')
    accounting=module('retained_accounting',APP/'collected/runtime/campaign_processes.py')
    manifest=read(APP/'collected/manifests/current-parakeet.json')
    results={};phases={};resources=[]
    check_run,_=original_checker()
    check_run(state['runs'][0],state,folder,spec,built,expected,protocol,accounting,manifest,results,phases,resources)
    return dict(refusal=refusal,state=state,spec=spec,built=built,receipt=receipt,
        expected=expected,protocol=protocol,accounting=accounting,manifest=manifest,
        results=results,resources=resources)


if __name__=='__main__':
    import json
    value=partial()
    print(json.dumps(dict(refusal=value['refusal'],requests=len(value['results']['control']['records']),resources=value['resources'])))
