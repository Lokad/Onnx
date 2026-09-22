"""Close results on the VM while the workstation cannot retain a full collection."""
from run import PRELUDE, ssh


script=PRELUDE+'''
from protocol import JOBS,LIMITS,check_sample,pin,read,save,verify
from checks import check_result,check_native
from remote import live
import numpy as np
assert not (base/'remote-closed.json').exists()
payload=verify(base);state=read(base/'identity.json');receipt=read(base/'collection.json')
assert state['complete'] and state['code']==0 and receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
assert state['supervisor']==read(base/'deployment.json') and state['boot_time']==1789634288.0
assert [r['name'] for r in state['runs']]==JOBS
ids=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
assert receipt['identities']==ids and not any(live(i) for i in ids)
for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
reports={'selected-256':check_result(read(base/'retained-256/result.json'),'selected','256',payload,base,base/'retained-256')}
resources=[]
for r in state['runs']:
 assert r['complete'] and r['code']==0 and r['seconds']<900
 assert r['preflight']['available']>=12*1024**3 and r['preflight']['tmpfs']>=3*1024**3
 samples=[json.loads(s) for s in (base/'logs'/(r['name']+'.jsonl')).read_text().splitlines()]
 assert len(samples)==r['samples'] and max(v['rss'] for v in samples)==r['peak_rss']
 for v in samples:
  check_sample(v)
  assert all(r['members'][str(m['pid'])]==m['birth'] for m in v['members'])
 result=read(base/r['name']/'result.json');assert result['pid']==r['child']['pid']
 mode,width=r['name'].split('-')
 if mode=='native':reports[r['name']]=check_native(result,base/r['name'],base)
 else:
  assert result['runtime']=='10.0.8'
  reports[r['name']]=check_result(result,mode,width,payload,base,base/r['name'])
 resources.append(dict(name=r['name'],samples=len(samples),peak_rss=r['peak_rss'],seconds=r['seconds']))
native=read(base/'native-ort/result.json');comparisons={};agreement={}
for width in ['256','512','scalar']:
 folder=base/'retained-256' if width=='256' else base/('selected-'+width)
 maximum=0.;count=0
 for row in native['reports']:
  a=np.fromfile(folder/row['file'],dtype='<f4').astype('float64');b=np.fromfile(base/'native-ort'/row['file'],dtype='<f4').astype('float64')
  assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
  error=float((np.abs(a-b)/np.maximum(1.,np.abs(b))).max(initial=0));assert error<=1e-4
  maximum=max(maximum,error);count+=int(a.size)
 assert count==1815552
 comparisons[width]=dict(values=count,maximum=maximum)
 agreement[width]=all(pin(folder/r['file'])==pin(base/'retained-256'/r['file']) for r in native['reports'])
assert state['ended']-state['started']<4*3600
verify(base)
analysis=dict(passed=True,reports=reports,resources=resources,native_comparisons=comparisons,width_agreement=agreement,
 core=payload['cores']['selected'],consumer=payload['consumer'],no_performance_measurement=True,
 candidate_admission_pending=True,local_collection_pending=True)
save(base/'remote-analysis.json',analysis)
files={p.relative_to(base).as_posix():pin(p) for p in base.rglob('*') if p.is_file() and p.name!='transfer.tar.gz'}
save(base/'remote-closed.json',dict(passed=True,files=files,terminal_identities=ids,external=payload['external'],analysis=pin(base/'remote-analysis.json')))
print(json.dumps(dict(closed=pin(base/'remote-closed.json'),**analysis)))
'''
envelope='from pathlib import Path\nsource='+repr(script)+'\nPath("/dev/shm/lokad-pyannote-lstm-platform-reference-v3-20260922/audit_complete.py").write_text(source,encoding="utf8")\nexec(compile(source,"remote-lstm-audit","exec"))\n'
print(ssh(envelope))
