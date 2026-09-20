"""Read the existing supervisor and worker state without starting any workload."""
from deploy import ssh,REMOTE

print(ssh('''from pathlib import Path
import json,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(%r);state=json.loads((base/'campaign/identity.json').read_text())
def live(b):
 try:return psutil.Process(b['pid']).create_time()==b['birth']
 except psutil.NoSuchProcess:return False
result=dict(complete=state['complete'],code=state['code'],error=state.get('error'),supervisor=state['supervisor'],supervisor_live=live(state['supervisor']),available=psutil.virtual_memory().available,runs=[])
for run in state['runs']:
 folder=base/run['output'];rows=sorted((folder/'worker').glob('[0-9][0-9][0-9].json'))
 value=dict(name=run['name'],complete=run['complete'],code=run['code'],error=run.get('error'),calls=len(rows),samples=run['samples'],peak_rss=run['peak_rss'],child=run.get('child'))
 if run.get('child'):value['child_live']=live(run['child'])
 if rows:
  row=json.loads(rows[-1].read_text());value.update(last_name=row['name'],last_allocated=row['allocated_bytes'],last_heap=row['memory_after']['managed_estimate'],last_gc=row['gc_after'],last_pools=row['pools'])
 result['runs'].append(value)
if (base/'campaign/conformance-gate.json').exists():result['gate']=json.loads((base/'campaign/conformance-gate.json').read_text())
if (base/'supervisor.stderr').exists():result['stderr_tail']=(base/'supervisor.stderr').read_text()[-2000:]
print(json.dumps(result))
'''%REMOTE))
