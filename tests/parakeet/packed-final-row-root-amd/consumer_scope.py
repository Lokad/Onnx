"""Preserve existing root/package checks, adding only the qualified test census."""
from pathlib import Path
import re
from protocol import pin
from new_cases import NEW_CASES,SKIPPED_CASES,FRAMES,FACTS,OWNED_CASES
from public_tests import qualified_census,verify as verify_public

TOOLS=Path(__file__).resolve().parent;ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'slice-dense-conversion-root-amd'


def verify_scope():
    files={}
    for name in ['protocol.py','remote.py','admission.py','audit.py','warning_census.py','graph_prerequisite.py']:
        source=ORIGINAL/name;assert (TOOLS/name).read_bytes()==source.read_bytes(),name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    source=ORIGINAL/'checks.py'
    expected=source.read_text().replace('from new_cases import NEW_CASES','from new_cases import NEW_CASES,SKIPPED_CASES')
    expected=expected.replace('3254','3277').replace('(3499,41)','(3539,42)').replace('(3409,131)','(3449,132)')
    expected=expected.replace("Counter((test,'Passed') for test in NEW_CASES[name])",
        "Counter((test,'NotExecuted' if test in SKIPPED_CASES[name] else 'Passed') for test in NEW_CASES[name])")
    assert (TOOLS/'checks.py').read_text()==expected
    files[source.relative_to(ROOT).as_posix()]=pin(source)
    qualified=ROOT/'artifacts/parakeet-slice-dense-conversion-tests-amd-20260924/bundle/source/tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'
    assert (TOOLS/'SliceDenseConversionTests.cs.txt').read_bytes()==qualified.read_bytes()
    assert (ROOT/'artifacts/parakeet-packed-final-row-source-20260925/source/tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs').read_bytes()==qualified.read_bytes()
    draft=qualified.read_text()
    assert list(map(int,re.search(r'Frames => new\[\]\s*\{([^}]+)\}',draft).group(1).split(',')))==FRAMES
    assert sorted(re.findall(r'\[Fact\]\s+public void (\w+)\(',draft))==sorted(FACTS)
    assert draft.count('[Theory]')==1 and len(NEW_CASES['tensors'])==26 and not SKIPPED_CASES['tensors']
    files[qualified.relative_to(ROOT).as_posix()]=pin(qualified)
    census=qualified_census()
    assert OWNED_CASES==census['passed'] and SKIPPED_CASES['backend']==census['skipped']
    assert NEW_CASES['backend']==sorted(census['passed']+census['skipped'])
    original=ROOT/'artifacts/parakeet-packed-final-row-source-20260925/source/tests/Lokad.Onnx.Backend.Tests/OwnedPackedWeightTests.cs'
    verify_public(original.read_text(encoding='utf8'),(TOOLS/'OwnedPackedWeightTests.cs.txt').read_text(encoding='utf8'))
    files[original.relative_to(ROOT).as_posix()]=pin(original)
    return files


if __name__=='__main__':print(verify_scope())
