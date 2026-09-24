"""Require every original and new tensor case in both consumed CPU modes."""
import json
import xml.etree.ElementTree as ET
from run import BASE,pin,read,write,prepared
from checks import resources


def main():
    prepared();assert not (BASE/'closed.json').exists()
    resource_rows=resources('capture');folder=BASE/'capture-collected';spec=read(BASE/'bundle/spec.json')
    built=read(BASE/'build-collected/built.json');review=read(BASE/'build-review.json')
    assert review['passed'] and review['built']==pin(BASE/'build-collected/built.json')
    assert pin(folder/'built.json')==pin(BASE/'build-collected/built.json')
    state=read(folder/'capture-state.json');assert [r['name'] for r in state['runs']]==['tensors-512','tensors-256']
    ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'};suites=[];names=None
    for mode in ['512','256']:
        path=folder/'logs'/('tensors-'+mode+'.trx');tree=ET.parse(path)
        summary=tree.find('.//t:ResultSummary',ns);assert summary is not None and summary.attrib['outcome']=='Completed'
        counters=summary.find('t:Counters',ns);assert counters is not None
        assert int(counters.attrib['total'])==int(counters.attrib['executed'])==int(counters.attrib['passed'])==spec['expected_tests']==395
        assert int(counters.attrib['failed'])==int(counters.attrib['notExecuted'])==0
        rows=tree.findall('.//t:UnitTestResult',ns);assert len(rows)==395 and all(r.attrib['outcome']=='Passed' for r in rows)
        current=sorted(r.attrib['testName'] for r in rows);assert len(set(current))==len(current)
        if names is None:names=current
        else:assert current==names
        assert sum('SliceDenseConversionTests.' in n for n in current)==26
        assert sum('DenseCopyCandidateIdentityTests.ConsumedCoreAndInstructionModeMatch' in n for n in current)==1
        suites.append(dict(mode=mode,passed=len(rows),skipped=0,trx=pin(path)))
    analysis=dict(passed=True,core=built['core'],consumer=built['consumer'],suites=suites,resources=resource_rows,
        new_copy_cases=26,existing_tensor_cases=368,identity_cases=1,original_methods_unchanged=3253,
        intentional_inherited_override=True,performance_measured=False)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),build_review=pin(BASE/'build-review.json'),
        collection=pin(folder/'capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),auditor=pin(__file__),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]))
    print(json.dumps(analysis))


if __name__=='__main__':main()
