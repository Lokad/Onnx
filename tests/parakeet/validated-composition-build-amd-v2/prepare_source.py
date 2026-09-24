"""Correct explicit test arguments while preserving every product source byte."""
import difflib
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
loader=importlib.util.spec_from_file_location('original_composition',Path(__file__).resolve().parents[1]/'validated-composition-build-amd/prepare_source.py')
parent=importlib.util.module_from_spec(loader);loader.loader.exec_module(parent)
pin,read,APPS=parent.pin,parent.read,parent.APPS
PRIOR=ROOT/'artifacts/parakeet-validated-composition-source-20260924'
FAILURE=ROOT/'artifacts/parakeet-validated-composition-build-amd-20260924/closed.json'
SOURCE=ROOT/'artifacts/parakeet-validated-composition-source-v2-20260924'
TEST='tests/Lokad.Onnx.Backend.Tests/PreparedLstmWeightsTests.cs'


def inspect():
    parent.inspect()
    assert pin(PRIOR/'prepared.json')['sha256']=='2000c839be501cdeb4f274f6e6ffda7652fad33aecbb0f43ecd51b28875dce34'
    assert pin(FAILURE)['sha256']=='a51a8a8dc4d7cac32b16c61abebfc17b09bc954544d4443821aabee66a4236d6'
    failure=read(FAILURE);assert failure['terminal'] and not failure['passed'] and failure['passed_tests']==368
    assert failure['failed_tests']==['Lokad.Onnx.Tensors.Tests.NoOptionalParametersTests.SourceTree_HasNoOptionalParameters']
    source=read(PRIOR/'prepared.json');assert source['passed']
    for name,wanted in source['source'].items():assert pin(PRIOR/'source'/name)==wanted,name
    for label,folder in APPS.items():
        assert source['admissions'][label]['closure']==pin(folder/'closed.json') and read(folder/'closed.json')['admitted']
    return source


def main():
    source=inspect();assert not sys.argv[1:] and not SOURCE.exists()
    original=(PRIOR/'source'/TEST).read_text(encoding='utf8')
    edits=[('Graph(long budget = PairBytes)','Graph(long budget)'),
        ('Run(ComputationalGraph graph, ExecutionOptions? options = null)','Run(ComputationalGraph graph, ExecutionOptions? options)'),
        ('Graph()','Graph(PairBytes)'),('Run(graph)','Run(graph, null)')]
    counts=[original.count(before) for before,_ in edits];assert counts==[1,1,4,9],counts
    assert all(after not in original for before,after in edits)
    changed=original
    for before,after in edits:changed=changed.replace(before,after)
    restored=changed
    for before,after in reversed(edits):restored=restored.replace(after,before)
    assert restored==original
    SOURCE.mkdir();snapshot=SOURCE/'source';snapshot.mkdir()
    for name,wanted in source['source'].items():
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True)
        if name==TEST:target.write_text(changed,encoding='utf8')
        else:target.write_bytes((PRIOR/'source'/name).read_bytes());assert pin(target)==wanted,name
    patch=''.join(difflib.unified_diff(original.splitlines(True),changed.splitlines(True),fromfile='a/'+TEST,tofile='b/'+TEST))
    (SOURCE/'test-arguments.patch').write_text(patch,encoding='utf8')
    identities={name:pin(snapshot/name) for name in source['source']}
    assert [name for name in identities if identities[name]!=source['source'][name]]==[TEST]
    assert all(identities[name]==source['source'][name] for name in identities if name.startswith('src/'))
    value=dict(source,source=identities,generator=pin(Path(__file__)),all_parent_bytes_preserved=False,
        all_product_parent_bytes_preserved=True,product_source_prepared=pin(PRIOR/'prepared.json'),
        previous_failure=pin(FAILURE),corrected_test=TEST,corrected_test_arguments=dict(Graph=4,Run=9),
        test_correction_patch=pin(SOURCE/'test-arguments.patch'),
        correction='Remove two optional declarations and pass their original default values explicitly at every call site')
    with (SOURCE/'prepared.json').open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(passed=True,prepared=pin(SOURCE/'prepared.json'),source_files=425,product_sources_unchanged=True,corrected_test=TEST,call_sites=13)))


if __name__=='__main__':main()
