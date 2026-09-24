"""Correct one retained-result path without changing the frozen capture or checks."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
TOOLS=ROOT/'tests/parakeet/masking-padding-layout-amd'
sys.path.insert(0,str(TOOLS))
from run import BASE,APP,pin,read,write,prepared


def correction():
    original=TOOLS/'audit.py'
    assert pin(original)==read(BASE/'prepared.json')['tools']['audit.py']
    before='collected/timing-01-candidate/result.json'
    after='collected/timing-01-candidate/output/result.json'
    assert not (APP/before).exists()
    proof=read(APP/'closed.json');assert proof['passed'] and proof['admitted']
    assert pin(APP/'closed.json')['sha256']=='f04c09fbc6c0455c4420d6680d60bb4f8fc5cac9dda2f2507768ba94b86335e4'
    assert pin(APP/after)==proof['files'][after]
    code=original.read_text(encoding='utf8');assert code.count(before)==1
    code=code.replace(before,after)
    needle="write(BASE/'closed.json',dict(passed=True,"
    assert code.count(needle)==1
    code=code.replace(needle,"write(BASE/'closed.json',dict(passed=True,audit_correction=pin(BASE/'audit-correction.json'),")
    value=dict(passed=True,original_auditor=pin(original),corrected_auditor=pin(Path(__file__)),
        previous_missing_path=before,retained_reference_path=after,reference=pin(APP/after),
        application_closure=pin(APP/'closed.json'),new_inference_calls=0,
        scope='Only the retained reference path and an explicit closure correction pin change; every original check remains.')
    return code,value


def main():
    prepared();assert not (BASE/'closed.json').exists() and not (BASE/'audit-correction.json').exists()
    code,value=correction();write(BASE/'audit-correction.json',value)
    namespace=dict(__name__='corrected_layout_audit',__file__=str(Path(__file__).resolve()))
    exec(compile(code,str(TOOLS/'audit.py')+' [retained path corrected]','exec'),namespace)
    namespace['main']()


if __name__=='__main__':main()
