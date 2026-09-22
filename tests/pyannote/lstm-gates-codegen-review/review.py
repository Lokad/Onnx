"""Record complete helper arithmetic and direct activation calls from actual JIT bodies."""
import hashlib
import json
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/pyannote-lstm-gates-codegen-amd-20260923'
OUT=ROOT/'artifacts/pyannote-lstm-gates-codegen-review-20260923'
def read(p):return json.loads(p.read_text())
def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
assert not OUT.exists()
assert pin(BASE/'closed.json')['sha256']=='9daee36185614fb6a4c5a5d2993b5484941a36b704efa5efae511eb748325d73'
for name,wanted in read(BASE/'closed.json')['files'].items():assert pin(BASE/name)==wanted,name
listings=read(BASE/'listings.json');rows={}
for mode in ['256','512','scalar','simd']:
    helper,=[b for b in listings['candidate-'+mode] if ':LstmUpdateDefaultGates(' in b['method'] and b['tier']=='Tier1' and b['complete_uninterleaved']]
    calls=re.findall(r'call\s+System.MathF:(Exp|Tanh)\(float\):float',helper['body'])
    assert calls==['Exp','Tanh','Exp','Exp','Tanh']
    assert 'System.Func' not in helper['body'] and 'CORINFO_HELP_NEWSFAST' not in helper['body']
    assert not re.search(r'\bv(?:f|fn)m(?:add|sub)\d*(?:ps|ss)\b',helper['body'])
    rows[mode]=dict(bytes=helper['code_bytes'][0],activation_calls=calls,
        scalar_multiplies=len(re.findall(r'\bv?mulss\b',helper['body'])),scalar_adds=len(re.findall(r'\bv?addss\b',helper['body'])),
        scalar_vector_register_stack_references=[line.strip() for line in helper['body'].splitlines() if 'xmm' in line and re.search(r'\[(?:rbp|rsp)',line)],
        recovered_prefix=helper.get('recovered_prefix',False))
    assert rows[mode]['scalar_multiplies']==3 and rows[mode]['scalar_adds']==16
OUT.mkdir();value=dict(passed=True,codegen=pin(BASE/'closed.json'),reviewer=pin(Path(__file__)),modes=rows,
    manual_review='The body retains three separately rounded additions per gate, three exp/two tanh calls in selected order, separate cell multiplies/add and output multiply, then writes owned state/output spans. Scalar values spill across calls; no delegate dispatch or fused arithmetic. Generic update remains exact in source and focused unbounded-clip tests exercise it.',
    no_performance_claim=True)
(OUT/'review.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(dict(review=pin(OUT/'review.json'),modes={m:{k:v for k,v in r.items() if k!='scalar_vector_register_stack_references'} for m,r in rows.items()})))
