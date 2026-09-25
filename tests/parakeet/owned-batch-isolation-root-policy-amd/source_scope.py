"""Bind the measured source plus exactly the four documented policy corrections."""
import importlib.util
from pathlib import Path
from protocol import pin,read

ROOT=Path(__file__).resolve().parents[3]
TOOLS=Path(__file__).resolve().parent
ORIGINAL=TOOLS.parent/'owned-batch-isolation-root-amd'
CORRECTION=ROOT/'artifacts/parakeet-owned-batch-isolation-root-policy-20260925'
FAILED=ROOT/'artifacts/parakeet-owned-batch-isolation-root-amd-20260925'


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


parent=load('retained_root_source_scope',ORIGINAL/'source_scope.py')
SOURCE,OWNED_PUBLIC,DEPTHWISE_PUBLIC=parent.SOURCE,parent.OWNED_PUBLIC,parent.DEPTHWISE_PUBLIC
TEMPLATES,TEST,DEPTHWISE=parent.TEMPLATES,parent.TEST,parent.DEPTHWISE
verify_templates=parent.verify_templates
# Retained guards test the original 427-to-435 candidate source transition.
# The current four-file policy delta is checked separately by correction().
CHANGED,ADDED,delta=parent.CHANGED,parent.ADDED,parent.delta
GENERATOR=TOOLS.parent/'owned-batch-isolation-root-correction/correct_source.py'


def correction():
    receipt=read(CORRECTION/'applied.json')
    assert receipt['passed'] and receipt['policy_guards_unchanged'] and receipt['generator']==pin(GENERATOR)
    assert receipt['failed_root']==pin(FAILED/'closed.json')
    assert receipt['failed_root']['sha256']=='f129f7d80bf1013d39bcd69778244803e2547c57b2c03d01f971d4f344ec5075'
    failed=read(FAILED/'closed.json');assert not failed['passed'] and failed['preserved_failure']
    for name,wanted in failed['files'].items():assert pin(FAILED/name)==wanted,name
    helper=load('retained_policy_correction',GENERATOR)
    sources,calls=helper.corrected_sources()
    assert receipt['calls']==calls and len(calls)==49 and set(receipt['after'])==set(sources)
    for name,contents in sources.items():
        assert (CORRECTION/'source'/name).read_bytes()==contents
        assert pin(CORRECTION/'source'/name)==receipt['after'][name]
        assert pin(CORRECTION/'before'/name)==receipt['before'][name]
        assert receipt['before'][name]==pin(FAILED/'bundle/source'/name)
    return receipt


def verify_source():
    source=parent.verify_source();correction()
    return source


def root_files(source):
    result=parent.root_files(source);policy=correction()
    result.update(policy['after']);assert len(result)==435
    assert result['src/Lokad.Onnx/Global.cs']==source['before']['src/Lokad.Onnx/Global.cs']
    for name in ['tests/Lokad.Onnx.Tensors.Tests/ImplementorDocsTests.cs',
                 'tests/Lokad.Onnx.Tensors.Tests/NoOptionalParametersTests.cs']:
        assert result[name]==pin(FAILED/'bundle/source'/name)
    return result


if __name__=='__main__':
    source=verify_source();files=root_files(source)
    for name,wanted in files.items():assert pin(ROOT/name)==wanted,name
    print(dict(passed=True,files=len(files),corrected_paths=4,method_bodies_changed=0,policy_guards_changed=0))
