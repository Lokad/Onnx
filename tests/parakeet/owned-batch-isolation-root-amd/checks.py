"""Exact source-equivalent product, complete test census and real PackageReference checks."""
from collections import Counter
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
from protocol import pin,read
from new_cases import NEW_CASES,SKIPPED_CASES

HARDWARE='PackedAvx512RowTests.RowGroupsMatchIndependentFmaOracleAndPreserveGuards'


def inventory(value,measured,built):
    assert value['inventory_complete'] and len(value['observations'])==2
    for row,(name,count) in zip(value['observations'],[('Lokad.Onnx.dll',3281),('Lokad.Onnx.Data.dll',697)]):
        assert row['assembly']==name and row['methods']==row['unchanged_methods']==count
        assert row['public_surface_equal'] and row['compiler_rename'] is None
        assert not row['differences'] and not row['added'] and not row['removed']
        assert row['before_sha256']==measured[name]['sha256'] and row['after_sha256']==built[name]['sha256']
        assert len(row['normalized_methods'])==count and not row['candidate_methods']
        assert row['method_flags_before']==row['method_flags_after'] and len(row['method_flags_after'])==count
    return dict(passed=True,core_methods=3281,data_methods=697,public_surface_equal=True,implementation_flags_equal=True)


def census(path):
    doc=ET.parse(path);rows=doc.findall('.//{*}UnitTestResult');counts=doc.find('.//{*}Counters').attrib
    assert len(rows)==int(counts['total']) and int(counts['failed'])==0
    assert all(r.attrib['outcome'] in ['Passed','NotExecuted'] for r in rows)
    assert sum(r.attrib['outcome']=='Passed' for r in rows)==int(counts['passed'])
    return Counter((r.attrib['testName'],r.attrib['outcome']) for r in rows)


def suite(path,name,evidence):
    actual=census(path);expected=census(evidence/('selected-'+name+'.trx'))
    additions=Counter((test,'NotExecuted' if test in SKIPPED_CASES[name] else 'Passed') for test in NEW_CASES[name])
    assert not set(test for test,_ in expected) & set(NEW_CASES[name])
    expected.update(additions)
    assert actual==expected,dict(missing=list((expected-actual).items())[:15],extra=list((actual-expected).items())[:15])
    passed=sum(n for (_,outcome),n in actual.items() if outcome=='Passed');skipped=sum(actual.values())-passed
    assert (passed,skipped)==((3546,42) if name=='backend' else (394,0))
    if name=='backend':assert any(HARDWARE in test and outcome=='Passed' for test,outcome in actual)
    return dict(passed=passed,skipped=skipped,census_exact=True,trx=pin(path))


def package(path,core):
    with zipfile.ZipFile(path) as z:
        data=z.read('lib/net10.0/Lokad.Onnx.dll')
        assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==core
        deps=[n.attrib for n in ET.fromstring(z.read('Lokad.Onnx.nuspec')).iter() if n.tag.split('}')[-1]=='dependency']
        assert len(deps)==1 and deps[0]['id']=='Google.Protobuf' and deps[0]['version']=='3.33.5'
    return dict(passed=True,package=pin(path),core=core,dependencies=deps)


def consumer(value,core,executable,model):
    assert value['passed'] and value['core']==core['sha256'] and value['executable']==executable['sha256'] and value['model']==model['sha256']
    assert value['processor_count']==1 and value['runtime']=='.NET 10.0.8'
    assert value['input_and_held_outputs_unchanged'] and value['model_imported']
    assert value['product']==[5,11,14,23] and value['convolution']==[-3,3,-7,7] and value['activated']==[0,3,0,7]
    assert value['prepared_graph_values']==1056 and value['prepared_graph_calls']==2 and value['retained_weights']==18432 and value['graph_scratch']==8384
    assert (value['winograd_values'],value['winograd_calls'],value['winograd_weights'],value['winograd_scratch'])==(1056,2,51200,28800)
    return dict(passed=True,winograd_values=1056,winograd_calls=2,winograd_weights=51200,winograd_scratch=28800,prepared_graph_values=1056,prepared_graph_calls=2,retained_weights=18432,scratch=8384,ownership=True)


def suite256(path,name,evidence):
    actual=census(path);expected=census(evidence/('selected-'+name+'.trx'))
    additions=Counter((test,'NotExecuted' if test in SKIPPED_CASES[name] else 'Passed') for test in NEW_CASES[name])
    assert not set(test for test,_ in expected) & set(NEW_CASES[name])
    expected.update(additions)
    if name=='backend':
        inactive=['PackedAvx512RowTests.RowGroupsMatchIndependentFmaOracleAndPreserveGuards',
            'PackedAvx512NarrowTests.RemaindersPreserveCompleteBitsAccumulationAndOwnership',
            'PackedPanelTraversalTests.ComposerMatchesOriginalBitsAndIndependentCoordinates',
            'MatMulKernelAgreementTests.SixRow512MatchesTiledBitwise',
            'MatMulKernelAgreementTests.SixRow512AlphaMatchesTiledAlphaBitwise',
            'Exp512Tests.ProbeHoldsContractOnWideSweep',
            'Exp512Tests.ProbeHoldsContractOnReducedRange',
            'Exp512Tests.ProbeMatchesEstrinCore',
            'Exp512Tests.ProbeHandlesExceptionals',
            'Exp512Tests.ProbeTailsMatchScalar']
        active=['PackedAvx512RowTests.UnsupportedHardwareDeclinesWithoutTouchingMemory',
            'PackedAvx512NarrowTests.UnsupportedHardwareDeclinesBeforeReadingPointers',
            'PackedPanelTraversalTests.UnsupportedHardwareDeclinesBeforeReadingPointers']
        disabled=enabled=0
        for (test,outcome),count in list(expected.items()):
            if any(test.startswith('Lokad.Onnx.Backend.Tests.'+method) for method in inactive):
                assert outcome=='Passed';del expected[(test,outcome)];expected[(test,'NotExecuted')]+=count;disabled+=count
            if any(test=='Lokad.Onnx.Backend.Tests.'+method for method in active):
                assert outcome=='NotExecuted';del expected[(test,outcome)];expected[(test,'Passed')]+=count;enabled+=count
        assert (disabled,enabled)==(93,3)
    assert actual==expected,dict(missing=list((expected-actual).items())[:15],extra=list((actual-expected).items())[:15])
    passed=sum(n for (_,outcome),n in actual.items() if outcome=='Passed');skipped=sum(actual.values())-passed
    assert (passed,skipped)==((3456,132) if name=='backend' else (394,0))
    return dict(passed=passed,skipped=skipped,census_exact=True,avx512_disabled=True,trx=pin(path))
