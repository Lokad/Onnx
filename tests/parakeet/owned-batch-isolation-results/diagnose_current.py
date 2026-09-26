"""Correct only the mislabeled mapping-receipt pin in the frozen analyzer."""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
TOOLS=ROOT/'tests/parakeet/owned-batch-isolation-profile-resume-amd'
sys.path.insert(0,str(TOOLS))
from resume import BASE, pin, read

source=TOOLS/'diagnose.py'
assert pin(source)==read(BASE/'prepared.json')['inputs'][source.relative_to(ROOT).as_posix()]
text=source.read_text(encoding='utf8')
before="assert pin(OLD_GAP/'analysis.json')['sha256'] == '17f0e96862a89606b7e6f56ac5b3ba13fcd36507efa7388bf8c15fff5e0899a7'"
after=before.replace("OLD_GAP/'analysis.json'","OLD_GAP/'closed.json'")
assert text.count(before)==1
# The immediately preceding assertion still binds the analysis to this closure.
assert "old_proof['analysis'] == pin(OLD_GAP/'analysis.json')" in text
text=text.replace(before,after)
reviewer="reviewer=pin(TOOLS/'diagnose.py')"
assert text.count(reviewer)==1
text=text.replace(reviewer,"reviewer=pin(Path(__file__)), frozen_analyzer=pin(TOOLS/'diagnose.py'), mapping_pin_correction=True")
scope=dict(__name__='corrected_current_attribution',__file__=__file__,Path=Path)
exec(compile(text,str(source)+' [mapping receipt identity corrected]','exec'),scope)

if __name__=='__main__': scope['main']()
