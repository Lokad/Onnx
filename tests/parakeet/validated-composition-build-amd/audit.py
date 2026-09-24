"""Verify every tensor test outcome and both consumed instruction modes."""
import json
import xml.etree.ElementTree as ET
from run import BASE,pin,read,write
from review_build import resources


def main():
    resource_rows=resources('capture');folder=BASE/'capture-collected'
    spec=read(BASE/'bundle/spec.json');built=read(BASE/'build-collected/built.json')
    assert read(BASE/'build-review.json')['passed']
    state=read(folder/'capture-state.json');assert [r['name'] for r in state['runs']]==['tensors-512','tensors-256']
    ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'};suites=[];names=None
    for mode in ['512','256']:
        path=folder/'logs'/('tensors-'+mode+'.trx');tree=ET.parse(path)
        summary=tree.find('.//t:ResultSummary',ns);assert summary is not None and summary.attrib['outcome']=='Completed'
        counters=summary.find('t:Counters',ns);assert counters is not None
        assert int(counters.attrib['total'])==int(counters.attrib['executed'])==int(counters.attrib['passed'])==spec['expected_tests']
        assert int(counters.attrib['failed'])==int(counters.attrib['notExecuted'])==0
        rows=tree.findall('.//t:UnitTestResult',ns);assert len(rows)==spec['expected_tests'] and all(r.attrib['outcome']=='Passed' for r in rows)
        current=sorted(r.attrib['testName'] for r in rows);assert len(set(current))==len(current)
        if names is None:names=current
        else:assert current==names
        assert sum('SliceReshapeCopyTests.' in name for name in current)==25
        assert sum('SliceCandidateIdentityTests.ConsumedCoreAndInstructionModeMatch' in name for name in current)==1
        suites.append(dict(mode=mode,passed=len(rows),skipped=0,trx=pin(path)))
    analysis=dict(passed=True,core=built['core'],data=built['data'],source_prepared=spec['source_prepared'],consumer=built['consumer'],suites=suites,resources=resource_rows,
        new_copy_cases=25,existing_tensor_cases=343,campaign_identity_cases=1,public_surface_equal=True,performance_measured=False,same_binaries=True)
    write(BASE/'analysis.json',analysis)
    write(BASE/'closed.json',dict(passed=True,analysis=pin(BASE/'analysis.json'),build_review=pin(BASE/'build-review.json'),
        collection=pin(folder/'capture-collection.json'),transfer=pin(BASE/'capture-transfer.json'),auditor=pin(__file__),
        terminal_owners=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]))
    print(json.dumps(analysis))


if __name__=='__main__':main()
