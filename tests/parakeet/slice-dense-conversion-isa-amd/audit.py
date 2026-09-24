"""Join the retained ordinary suite and correctly consumed disabled-mode suite."""
import json
import xml.etree.ElementTree as ET
from run import BASE,ORIGINAL,pin,read,write,prepared
from checks import resources


def main():
    prepared();assert not (BASE/'closed.json').exists()
    original_resources=resources('capture',original=True);resource_rows=resources('capture')
    folder=BASE/'capture-collected';state=read(folder/'capture-state.json');spec=read(BASE/'bundle/spec.json')
    assert state['instruction_environment']==spec['instruction_environment']=={'DOTNET_EnableAVX512':'0'}
    assert [r['name'] for r in state['runs']]==['tensors-256']
    ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'};names=None;suites=[]
    paths=[('512',ORIGINAL/'capture-collected/logs/tensors-512.trx'),('256',folder/'logs/tensors-256.trx')]
    for mode,path in paths:
        tree=ET.parse(path);summary=tree.find('.//t:ResultSummary',ns)
        assert summary.attrib['outcome']=='Completed';counters=summary.find('t:Counters',ns).attrib
        assert int(counters['total'])==int(counters['executed'])==int(counters['passed'])==395
        assert int(counters['failed'])==int(counters['notExecuted'])==0
        rows=tree.findall('.//t:UnitTestResult',ns);assert len(rows)==395 and all(r.attrib['outcome']=='Passed' for r in rows)
        current=sorted(r.attrib['testName'] for r in rows);assert len(set(current))==395
        if names is None:names=current
        else:assert names==current
        assert sum('SliceDenseConversionTests.' in n for n in current)==26
        assert sum('DenseCopyCandidateIdentityTests.ConsumedCoreAndInstructionModeMatch' in n for n in current)==1
        suites.append(dict(mode=mode,passed=395,skipped=0,trx=pin(path)))
    failed=ET.parse(ORIGINAL/'capture-collected/logs/tensors-256.trx').findall('.//t:UnitTestResult',ns)
    assert sorted(r.attrib['testName'] for r in failed)==names
    refusal,=[r for r in failed if r.attrib['outcome']!='Passed']
    assert refusal.attrib['testName']=='Lokad.Onnx.Tensors.Tests.DenseCopyCandidateIdentityTests.ConsumedCoreAndInstructionModeMatch'
    assert all(r.attrib['outcome']=='Passed' for r in failed if r is not refusal)
    analysis=dict(passed=True,core=spec['core'],consumer=spec['consumer'],suites=suites,resources=resource_rows,
        original_resources=original_resources,failed_instruction_run_retained=True,ordinary_run_repeated=False,
        corrected_test_review=pin(ORIGINAL/'build-review.json'),compiled_review=read(ORIGINAL/'build-review.json')['compiled_review'],
        original_methods_unchanged=3253,new_copy_cases=26,existing_tensor_cases=368,identity_cases=1,
        product_rebuilt=False,performance_measured=False)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),collection=pin(folder/'capture-collection.json'),
        original_collection=pin(ORIGINAL/'capture-collected/capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),
        auditor=pin(__file__),terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]))
    print(json.dumps(dict(passed=True,closed=pin(BASE/'closed.json'),core=spec['core'],suites=suites)))


if __name__=='__main__':main()
