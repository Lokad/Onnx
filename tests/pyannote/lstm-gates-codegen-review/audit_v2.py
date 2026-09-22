"""Recover one complete helper prefix by exact equality to an uninterleaved body.

Preserve the first failed audit, all raw listings and its deferred trailing
projection body. Do not rerun workers or relax the complete-body requirement.
"""
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/pyannote/lstm-gates-codegen-amd'))
import audit as original
from protocol import pin,read,save
BASE=original.BASE
raw_listings=original.listings
SIGNATURE='Lokad.Onnx.CPUExecutionProvider:LstmUpdateDefaultGates('

def instructions(body):
    start=body.index('G_M000_IG01:');end=re.search(r'; Total bytes of code \d+',body).end()
    return body[start:end]

def main():
    assert not (BASE/'closed.json').exists()
    state=read(BASE/'collected/identity.json');receipt=read(BASE/'collected/collection.json')
    assert state['complete'] and state['code']==0 and receipt['terminal'] and receipt['code']==0
    broken=raw_listings(BASE/'collected/logs/candidate-256.stdout')
    target,=[b for b in broken if b['method'].startswith(SIGNATURE) and b['tier']=='Tier1']
    assert target['code_bytes']==[670,924] and not target['complete_uninterleaved']
    clean,=[b for b in raw_listings(BASE/'collected/logs/candidate-512.stdout') if b['method']==target['method'] and b['tier']=='Tier1']
    assert clean['complete_uninterleaved'] and clean['code_bytes']==[670]
    assert instructions(target['body'])==instructions(clean['body'])
    footer=re.search(r'; Total bytes of code 670\n',target['body']);assert footer
    prefix=target['body'][:footer.end()];trailing=target['body'][footer.end():]
    assert trailing.lstrip().startswith('G_M000_IG01:') and '; Total bytes of code 924' in trailing
    assert len(re.findall(r'^G_M000_IG\d+:',prefix,re.M))==8
    if not (BASE/'failure-closed.json').exists():
        save(BASE/'failure-closed.json',dict(passed=False,worker_code=0,
            failure="Original audit: candidate-256 Missing optimized gate body. Parser attached the deferred projection body after the complete670byte helper, producing two code-size footers.",
            files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()},remote_terminal=receipt['identities']))
    else:
        for name,wanted in read(BASE/'failure-closed.json')['files'].items():assert pin(BASE/name)==wanted,name
    class Text:
        def read_text(self,encoding):return prefix
    recovered,=raw_listings(Text())
    assert recovered['complete_uninterleaved'] and recovered['code_bytes']==[670]
    recovered.update(line=target['line'],recovered_prefix=True,
        exact_clean_instruction_sha256=hashlib.sha256(instructions(clean['body']).encode()).hexdigest())
    def listings(path):
        rows=raw_listings(path)
        if path.name=='candidate-256.stdout':
            assert rows==broken
            return rows+[recovered]
        return rows
    save(BASE/'auditor-v2.json',dict(passed=True,tool=pin(Path(__file__)),original_auditor=pin(ROOT/'tests/pyannote/lstm-gates-codegen-amd/audit.py'),
        failure=pin(BASE/'failure-closed.json'),recovered=pin(BASE/'collected/logs/candidate-256.stdout'),
        clean_reference=pin(BASE/'collected/logs/candidate-512.stdout'),
        instructions_exact=True,code_bytes=670,original_fragments_retained=True,no_worker_rerun=True))
    original.listings=listings
    original.main()

if __name__=='__main__':main()
