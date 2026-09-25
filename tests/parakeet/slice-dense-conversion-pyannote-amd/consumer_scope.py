"""Preserve numerical/public/resource checks while reusing the exact qualified consumer."""
import ast
import hashlib
from pathlib import Path
from protocol import pin
TOOLS=Path(__file__).resolve().parent;ROOT=TOOLS.parents[2]
ORIGINAL=TOOLS.parent/'observed-dense-where-pyannote-amd'
REUSE_BLOCK_SHA='2f35ee909a309acf046380b136d40f4af127bb2f807d4a3a78b3ae30999216a0'


def verify_scope():
    files={}
    for name in ['checks.py','remote.py','candidate_protocol.py','qualify_outputs.py','test_semantics.py']:
        assert (TOOLS/name).read_bytes()==(ORIGINAL/name).read_bytes(),name
        files[(ORIGINAL/name).relative_to(ROOT).as_posix()]=pin(ORIGINAL/name)
    before=(ORIGINAL/'protocol.py').read_text();after=(TOOLS/'protocol.py').read_text()
    needle="JOBS = ['consumer-restore','consumer-build','consumer-inventory','selected','candidate']"
    assert before.count(needle)==1 and after==before.replace(needle,"JOBS = ['selected','candidate']")
    before=(ORIGINAL/'audit.py').read_text();after=(TOOLS/'audit.py').read_text()
    before=before.replace('APP_PAYLOAD,MODEL,OLD_DATA,NEW_DATA','APP_PAYLOAD,MODEL,CURRENT,NEW_DATA')
    before=before.replace("'M70 current release Core37c24375/Data cc37b19e' if role=='selected' else 'M70 observed-mask Coref95a13c5/Dataa893952f'","'M73 current release Coref95a13c5/Dataa893952f' if role=='selected' else 'M73 slice conversion Core49c3a958/Dataa893952f'")
    before=before.replace("(MODEL/'consumer/Program.cs').read_bytes().replace(OLD_DATA.encode(),NEW_DATA.encode())","(MODEL/'consumer/Program.cs').read_bytes()")
    marker="    built=read(collected/'built.json');";ending='    results={}'
    a,b=before.index(marker),before.index(ending)
    c,d=after.index(marker),after.index(ending)
    assert before[:a]==after[:c]
    assert before[b:].replace('inventory=il,results=results','inventory=il,consumer_reused_exactly=True,no_rebuild=True,results=results')==after[d:]
    assert hashlib.sha256(after[c:d].encode()).hexdigest()==REUSE_BLOCK_SHA
    for name in ['protocol.py','audit.py']:files[(ORIGINAL/name).relative_to(ROOT).as_posix()]=pin(ORIGINAL/name)
    return files


if __name__=='__main__':print(verify_scope())
