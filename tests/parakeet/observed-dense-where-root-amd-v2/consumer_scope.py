"""Preserve the complete existing root worker and each existing assertion."""
from pathlib import Path
import re
from protocol import pin
from new_cases import NEW_CASES,FRAMES,FACTS

TOOLS=Path(__file__).resolve().parent
ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'validated-composition-root-amd'


def verify_scope():
    files={}
    for name in ['protocol.py','remote.py','audit.py','admission.py','correction.py']:
        source=ORIGINAL/name
        assert (TOOLS/name).read_bytes()==source.read_bytes(),name
        files[source.relative_to(ROOT).as_posix()]=pin(source)
    source=ORIGINAL/'checks.py'
    expected=source.read_text().replace('3251','3253').replace('(3471,41)','(3499,41)').replace('(3381,131)','(3409,131)')
    assert (TOOLS/'checks.py').read_text()==expected
    files[source.relative_to(ROOT).as_posix()]=pin(source)
    graph=TOOLS.parent/'observed-dense-where-pyannote-app-amd/graph_prerequisite.py'
    assert (TOOLS/'graph_prerequisite.py').read_bytes()==graph.read_bytes()
    files[graph.relative_to(ROOT).as_posix()]=pin(graph)
    draft=(TOOLS/'DenseScalarWhereTests.cs.txt').read_text()
    frames=re.search(r'AttentionFrames => new\[\]\s*\{([^}]+)\}',draft).group(1)
    assert list(map(int,frames.split(',')))==FRAMES
    assert sorted(re.findall(r'\[Fact\]\s+public void (\w+)\(',draft))==sorted(FACTS)
    assert draft.count('[Theory]')==2 and draft.count('[InlineData(')==4
    assert len(NEW_CASES['backend'])==28 and NEW_CASES['tensors']==[]
    return files


if __name__=='__main__':print(verify_scope())
