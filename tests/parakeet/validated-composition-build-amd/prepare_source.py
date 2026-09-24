"""Compose only the exact sources of two independently admitted improvements."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
RECURRENCE=ROOT/'artifacts/parakeet-prepared-recurrence-source-20260924'
SLICE=ROOT/'artifacts/parakeet-slice-materialization-source-v2-20260924'
SOURCE=ROOT/'artifacts/parakeet-validated-composition-source-20260924'
APPS=dict(recurrence=ROOT/'artifacts/parakeet-prepared-recurrence-app-amd-20260924',
    slice=ROOT/'artifacts/parakeet-slice-materialization-app-amd-20260924')


def pin(path):
    with path.open('rb') as stream:return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def inspect():
    a=read(RECURRENCE/'prepared.json');b=read(SLICE/'prepared.json')
    assert pin(RECURRENCE/'prepared.json')['sha256']=='b52a89c1043165de1c376b37fc5307cd003a7c8b76f0f52508c4cdefcc669eab'
    assert pin(SLICE/'prepared.json')['sha256']=='d11cfc00158da39984da41da3cafc4ceb404d2f9e002c542a701f244d30a5150'
    assert a['passed'] and b['passed'] and a['before']==b['before'] and len(a['before'])==422
    for name,wanted in a['before'].items():assert pin(ROOT/name)==wanted,name
    changed={}
    for label,folder,record in [('recurrence',RECURRENCE,a),('slice',SLICE,b)]:
        for name,wanted in record['source'].items():assert pin(folder/'source'/name)==wanted,name
        changed[label]={name:wanted for name,wanted in record['source'].items() if record['before'].get(name)!=wanted}
    assert set(changed['recurrence']).isdisjoint(changed['slice'])
    assert set(changed['slice'])=={'src/Lokad.Onnx/TensorSlice.cs','tests/Lokad.Onnx.Tensors.Tests/SliceReshapeCopyTests.cs'}
    assert len(changed['recurrence'])==7
    expected=dict(a['source'],**changed['slice']);assert len(expected)==425
    return a,b,changed,expected


def main():
    a,b,changed,expected=inspect()
    if sys.argv[1:]==['inspect']:
        print(json.dumps(dict(compatible=True,source_files=len(expected),changed={k:list(v) for k,v in changed.items()})));return
    assert not sys.argv[1:] and not SOURCE.exists()
    admissions={}
    for label,folder in APPS.items():
        proof=read(folder/'closed.json');assert proof['passed'] and proof['admitted'],label
        assert proof['analysis']==pin(folder/'analysis.json')
        analysis=read(folder/'analysis.json');assert analysis['performance']['admitted']
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
        admissions[label]=dict(closure=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'),identities=analysis['identities'])
    assert admissions['recurrence']['identities']['current']==admissions['slice']['identities']['current']
    SOURCE.mkdir();snapshot=SOURCE/'source';snapshot.mkdir()
    for name,wanted in expected.items():
        source=(SLICE if name in changed['slice'] else RECURRENCE)/'source'/name
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes())
        assert pin(target)==wanted,name
    value=dict(passed=True,before=a['before'],source=expected,parents=dict(recurrence=pin(RECURRENCE/'prepared.json'),slice=pin(SLICE/'prepared.json')),
        admissions=admissions,modified=[name for name in a['before'] if expected[name]!=a['before'][name]],
        added=[name for name in expected if name not in a['before']],generator=pin(Path(__file__)),
        all_parent_bytes_preserved=True,root_product_changed=False,new_optimization=False)
    assert len(value['modified'])==6 and len(value['added'])==3
    with (SOURCE/'prepared.json').open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(prepared=pin(SOURCE/'prepared.json'),source_files=425,modified=value['modified'],added=value['added'])))


if __name__=='__main__':main()
