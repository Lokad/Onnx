"""Preserve full root/package contracts and bind the exact admitted public tests."""
from pathlib import Path
import re
from protocol import pin
from new_cases import NEW_CASES,FRAMES,FACTS

TOOLS=Path(__file__).resolve().parent;ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'observed-dense-where-root-amd-v2'


def verify_scope():
    files={}
    for name in ['protocol.py','remote.py','admission.py']:
        source=ORIGINAL/name;assert (TOOLS/name).read_bytes()==source.read_bytes(),name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    source=ORIGINAL/'audit.py'
    expected=source.read_text().replace('from checks import inventory,suite,package,consumer,suite256','from checks import inventory,suite,package,consumer,suite256\nfrom warning_census import compare as compare_warnings')
    expected=expected.replace('    analysis=dict(passed=True,root_source_verified=True,','    warnings=compare_warnings(collected)\n    analysis=dict(passed=True,warnings=warnings,root_source_verified=True,')
    assert (TOOLS/'audit.py').read_text()==expected
    files[source.relative_to(ROOT).as_posix()]=pin(source)
    source=ORIGINAL/'checks.py'
    assert (TOOLS/'checks.py').read_text()==source.read_text().replace('3253','3254').replace('(368,0)','(394,0)')
    files[source.relative_to(ROOT).as_posix()]=pin(source)
    graph=TOOLS.parent/'slice-dense-conversion-pyannote-app-amd/graph_prerequisite.py'
    assert (TOOLS/'graph_prerequisite.py').read_bytes()==graph.read_bytes()
    files[graph.relative_to(ROOT).as_posix()]=pin(graph)
    qualified=ROOT/'artifacts/parakeet-slice-dense-conversion-tests-amd-20260924/bundle/source/tests/Lokad.Onnx.Tensors.Tests/SliceDenseConversionTests.cs'
    assert (TOOLS/'SliceDenseConversionTests.cs.txt').read_bytes()==qualified.read_bytes()
    draft=qualified.read_text()
    assert list(map(int,re.search(r'Frames => new\[\]\s*\{([^}]+)\}',draft).group(1).split(',')))==FRAMES
    assert sorted(re.findall(r'\[Fact\]\s+public void (\w+)\(',draft))==sorted(FACTS)
    assert draft.count('[Theory]')==1 and len(NEW_CASES['tensors'])==26 and not NEW_CASES['backend']
    files[qualified.relative_to(ROOT).as_posix()]=pin(qualified)
    return files


if __name__=='__main__':print(verify_scope())
