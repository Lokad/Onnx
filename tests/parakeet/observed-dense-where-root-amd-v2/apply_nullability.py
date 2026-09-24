"""Apply only the source-policy repair after preserving the complete failed root."""
import difflib
import json
from prepare import ROOT,TOOLS,APPLIED,SOURCE,gates
from protocol import pin,read,save
from source_scope import root_files,TEST
from nullability import INITIAL,INCIDENT,HELPER,correct,verify_failure


def main():
    assert not APPLIED.exists()
    verify_failure();source=gates();initial=read(INITIAL/'applied.json')
    before=dict(source['source']);before[TEST]=pin(TOOLS/'DenseScalarWhereTests.cs.txt')
    assert initial['source_files']==before and len(before)==427
    for name,wanted in before.items():assert pin(ROOT/name)==wanted,name
    target=ROOT/HELPER
    assert target.resolve().is_relative_to(ROOT)
    original=target.read_bytes();corrected=correct(original)
    after=root_files(source)
    assert [name for name in after if after[name]!=before[name]]==[HELPER]
    APPLIED.mkdir()
    backup=APPLIED/'before'/HELPER;backup.parent.mkdir(parents=True);backup.write_bytes(original)
    patch=''.join(difflib.unified_diff(original.decode().splitlines(True),corrected.decode().splitlines(True),
        fromfile=HELPER+' (measured)',tofile=HELPER+' (nullable contract)'))
    (APPLIED/'source.patch').write_text(patch,encoding='utf8')
    correction=dict(failure=pin(INCIDENT/'closed.json'),original_integration=pin(INITIAL/'applied.json'),
        source_before=before[HELPER],source_after=after[HELPER],patch=pin(APPLIED/'source.patch'),
        transformer=pin(TOOLS/'nullability.py'),only_nullable_contract_changed=True,
        compiled_equivalence_pending=True)
    intended=dict(initial,source_files=after,source_correction=correction)
    save(APPLIED/'intended.json',intended)
    temporary=target.with_suffix('.cs.m70nullabletmp');assert not temporary.exists()
    temporary.write_bytes(corrected);assert pin(temporary)==after[HELPER];temporary.replace(target)
    for name,wanted in after.items():assert pin(ROOT/name)==wanted,name
    save(APPLIED/'applied.json',dict(intended,passed=True,root_build_pending=True))
    print(json.dumps(dict(passed=True,changed=HELPER,source_files=427,
        applied=pin(APPLIED/'applied.json'),compiled_equivalence_pending=True)))


if __name__=='__main__':main()
