"""Prepare one isolated Sigmoid edit over the exact measured M78 source."""
import difflib
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-vector-sigmoid-source-20260925'
PRIOR=ROOT/'artifacts/parakeet-packed-final-row-source-20260925'
TARGET='src/Lokad.Onnx/CPUExecutionProvider.Elementwise.cs'
TEST='tests/Lokad.Onnx.Backend.Tests/SigmoidVectorTests.cs'


def pin(path):return dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
def read(path):return json.loads(path.read_text(encoding='utf8'))
def write(path,value):
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def changed(original):
    start=original.index('    public static OpResult Sigmoid(')
    end=original.index('    /// <summary>Clip',start)
    before=original[start:end]
    replacement=(TOOLS/'Sigmoid.cs.txt').read_text(encoding='utf8').rstrip()+'\n\n'
    assert before.count('public static')==1 and before.count('MathF.Exp')==1
    assert before[before.index('            case TensorElementType.Double:'):]==replacement[replacement.index('            case TensorElementType.Double:'):]
    candidate=original[:start]+replacement+original[end:]
    assert candidate[:start]+before+candidate[start+len(replacement):]==original
    return candidate


def main():
    assert not BASE.exists()
    assert pin(PRIOR/'prepared.json')['sha256']=='e3ca64b50ea4a5dd276d90a19779d602f8ed78b6275b9e5448fe3b021bec5e1b'
    previous=read(PRIOR/'prepared.json');assert len(previous['source'])==433
    values={}
    for name,wanted in previous['source'].items():
        path=PRIOR/'source'/name;assert pin(path)==wanted,name;values[name]=path.read_bytes()
    gap=ROOT/'artifacts/parakeet-packed-final-row-gap-20260925'
    assert pin(gap/'closed.json')['sha256']=='17f0e96862a89606b7e6f56ac5b3ba13fcd36507efa7388bf8c15fff5e0899a7'
    assert read(gap/'closed.json')['analysis']==pin(gap/'analysis.json')
    before=values[TARGET].decode();after=changed(before)
    values[TARGET]=after.encode();assert TEST not in values;values[TEST]=(TOOLS/'SigmoidVectorTests.cs.txt').read_bytes()
    BASE.mkdir()
    for name,content in values.items():
        path=BASE/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(content)
    (BASE/'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=TARGET,tofile=TARGET)),encoding='utf8')
    (BASE/'prospective-plan.md').write_bytes((ROOT/'PLAN.md').read_bytes())
    source={n:pin(BASE/'source'/n) for n in values}
    assert [n for n,w in previous['source'].items() if source[n]!=w]==[TARGET]
    write(BASE/'prepared.json',dict(passed=True,root_product_changed=False,release_admitted=False,
        baseline=pin(PRIOR/'prepared.json'),gap=pin(gap/'closed.json'),source=source,
        changed_product_files=[TARGET],added_tests=[TEST],changed_methods=['CPUExecutionProvider.Sigmoid'],
        plan=pin(BASE/'prospective-plan.md'),patch=pin(BASE/'candidate.patch'),
        tools={p.name:pin(p) for p in TOOLS.iterdir() if p.is_file()}))
    print(json.dumps(dict(source=pin(BASE/'prepared.json'),files=len(source),changed_product_files=[TARGET])))


if __name__=='__main__':main()
