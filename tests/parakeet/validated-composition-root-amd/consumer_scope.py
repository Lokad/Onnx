"""Preserve the qualified worker and checks; add only the admitted test census."""
from pathlib import Path
import xml.etree.ElementTree as ET
from protocol import pin
from new_cases import NEW_CASES

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'wide-entry-first-use-root-amd-v2'


def verify_scope():
    files={}
    for name in ['protocol.py','remote.py','audit.py','admission.py','correction.py']:
        source=ORIGINAL/name
        assert (TOOLS/name).read_bytes()==source.read_bytes(),name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    source=ORIGINAL/'checks.py'
    expected=source.read_text().replace('from protocol import pin,read','from protocol import pin,read\nfrom new_cases import NEW_CASES').replace('3189','3251')
    before="actual=census(path);expected=census(evidence/('selected-'+name+'.trx'))"
    assert expected.count(before)==2
    expected=expected.replace(before,before+"\n    additions=Counter((test,'Passed') for test in NEW_CASES[name])\n    assert not set(test for test,_ in expected) & set(NEW_CASES[name])\n    expected.update(additions)")
    expected=expected.replace('(3449,41)','(3471,41)').replace('(3359,131)','(3381,131)').replace('(343,0)','(368,0)')
    assert (TOOLS/'checks.py').read_text()==expected
    files[source.relative_to(ROOT).as_posix()]=pin(source)
    fixtures={
        'backend':('artifacts/parakeet-prepared-recurrence-contracts-amd-20260924/collected/candidate-tests/contracts.trx','Lokad.Onnx.Backend.Tests.PreparedLstmWeightsTests.',22),
        'tensors':('artifacts/parakeet-validated-composition-build-amd-v2-20260924/capture-collected/logs/tensors-512.trx','Lokad.Onnx.Tensors.Tests.SliceReshapeCopyTests.',25)}
    assert set(NEW_CASES)==set(fixtures)
    for name,(relative,prefix,count) in fixtures.items():
        path=ROOT/relative
        rows=[r.attrib for r in ET.parse(path).findall('.//{*}UnitTestResult') if r.attrib['testName'].startswith(prefix)]
        assert len(rows)==count and all(r['outcome']=='Passed' for r in rows)
        assert sorted(r['testName'] for r in rows)==NEW_CASES[name]
        assert len(set(NEW_CASES[name]))==count
        files[relative]=pin(path)
    return files


if __name__=='__main__':print(verify_scope())
