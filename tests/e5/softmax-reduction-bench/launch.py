"""Launch one frozen phase; comparison additionally requires the audited control hash."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
base=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ('control','compare')
assert len(sys.argv)==(2 if phase=='control' else 3)
assert not (base/('deployment-'+phase+'.json')).exists() and not (base/('result-'+phase)).exists()
for rel,pin in json.loads((base/'bundle.json').read_text())['files'].items():
    p=base/rel;assert p.stat().st_size==pin['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'],rel
with (base/('supervisor-'+phase+'.log')).open('x') as log:
    child=subprocess.Popen(['taskset','-c','0','python3','-u',str(base/'run.py'),*sys.argv[1:]],cwd=base,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
line=Path('/proc',str(child.pid),'stat').read_text();fields=line[line.rfind(')')+2:].split()
deployment=dict(phase=phase,pid=child.pid,start=int(fields[19]),started=time.time())
with (base/('deployment-'+phase+'.json')).open('x') as f:json.dump(deployment,f,indent=2);f.write('\n')
print(json.dumps(deployment))
