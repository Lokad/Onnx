"""Audit the original product under the baseline-supported caller contract."""
from pathlib import Path
import sys
import acceptance

TOOLS=Path(__file__).resolve().parent
source=(TOOLS / 'audit.py').read_text(encoding='utf8')
changes=[('import successor','import acceptance as successor'),
    ("    schedule=[n+'-'+phase for n in ['cli','backend','tensors','bridge'] for phase in ['restore','build']]\n    schedule+=['inventory','caller-restore','caller-build','caller-normal','caller-disabled']",
     "    schedule=['caller-restore','caller-build','caller-normal','caller-disabled']"),
    ("        variants.append(dict(mode=mode,cases=len(rows),checked_values=sum(r['checked_values'] for r in rows),nan_values=sum(r['nan_values'] for r in rows)))",
     """        payloads=0
        for r in rows:
            assert len(r['baseline_digest'])==64
            for d in r['nan_payload_differences']:
                left,right=int(d['baseline'],16),int(d['candidate'],16)
                assert left!=right and all((v & 0x7f800000)==0x7f800000 and (v & 0x7fffff)!=0 for v in [left,right])
                assert d['batch'] in [0,1] and 3<=d['index']<r['checked_values']-3
            payloads+=len(r['nan_payload_differences'])
            if r['pattern']!='special':
                assert r['baseline_digest']==r['digest'] and not r['nan_payload_differences']
        variants.append(dict(mode=mode,cases=len(rows),checked_values=sum(r['checked_values'] for r in rows),
            nan_values=sum(r['nan_values'] for r in rows),nan_payload_differences=payloads))""")]
for old,new in changes:
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
namespace=dict(__name__='caller_acceptance_audit',__file__=str(TOOLS / 'audit.py'))
exec(compile(source,str(TOOLS / 'audit.py'),'exec'),namespace)
namespace['main']()
