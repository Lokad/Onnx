"""Repair only the zero-warning summary check; review the existing M78 build."""
import ast
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[3]
TOOLS=ROOT/'tests/parakeet/packed-final-row-profile-amd'
sys.path.insert(0,str(TOOLS))
from run import BASE,prepared,pin,read,write


def main():
    prepared()
    assert not (BASE/'build-review.json').exists()
    assert not (BASE/'build-review-correction.json').exists()
    original=TOOLS/'review_build.py'
    text=original.read_text(encoding='utf8')
    before="if 'warning' in line.lower()]"
    after="if 'warning' in line.lower() and line.strip() != '0 Warning(s)']"
    assert text.count(before)==1
    corrected=text.replace(before,after)
    assert corrected.replace(after,before)==text
    ast.parse(corrected)

    # Reproduce the actual false positive without executing the failed reviewer
    # again. Real diagnostics and nonzero summaries still fail the same check.
    logs=BASE/'build-collected/logs'
    rejected=[dict(log=p.name,text=line.strip()) for p in logs.iterdir()
        if p.suffix in ['.stdout','.stderr'] for line in p.read_text(encoding='utf8').splitlines()
        if 'warning' in line.lower()]
    assert sorted(rejected,key=lambda r:r['log'])==[
        dict(log='bridge-build.stdout',text='0 Warning(s)'),
        dict(log='data-build.stdout',text='0 Warning(s)')]
    condition=after.removesuffix(']').removeprefix('if ')
    for line in ['0 Warning(s)','    0 Warning(s)  ']:
        assert not eval(condition,{},dict(line=line))
    for line in ['1 Warning(s)','warning CS0001: failure','0 Warning(s) warning CS0001: failure']:
        assert eval(condition,{},dict(line=line))

    old=BASE/'build-review-original.py';new=BASE/'build-review-corrected.py'
    with old.open('xb') as stream:stream.write(original.read_bytes())
    with new.open('x',encoding='utf8',newline='\n') as stream:stream.write(corrected)
    receipt=dict(passed=True,original_reviewer=pin(original),preserved_original=pin(old),
        corrected_reviewer=pin(new),correction_tool=pin(__file__),
        exact_change='Exclude only the exact stripped 0 Warning(s) build summary',
        original_rejected_lines=rejected,
        logs={p.name:pin(p) for p in logs.iterdir() if p.suffix in ['.stdout','.stderr']},
        nonzero_summaries_and_diagnostics_still_rejected=True,
        rebuilt=False,inference_started=False,all_other_review_source_unchanged=True)
    write(BASE/'build-review-correction.json',receipt)
    runpy.run_path(str(new),run_name='__main__')
    review=read(BASE/'build-review.json')
    assert review['passed'] and review['reviewer']==pin(new) and not review['warnings']
    assert read(BASE/'build-review-transferred.json')['review']==pin(BASE/'build-review.json')
    print(json.dumps(dict(correction=pin(BASE/'build-review-correction.json'),review=pin(BASE/'build-review.json'))))


if __name__=='__main__':main()
