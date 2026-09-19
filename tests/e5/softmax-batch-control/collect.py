"""Collect a terminal phase exactly once, binding process births and every file."""
from pathlib import Path
import hashlib,json,subprocess,sys,tarfile,time
base=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ('control','compare')
archive=base.with_name(base.name+'-'+phase+'-results.tar.gz');assert not archive.exists()
deployment=json.loads((base/('deployment-'+phase+'.json')).read_text());identity=json.loads((base/('result-'+phase+'/identity.json')).read_text())
assert deployment['phase']==identity['phase']==phase and deployment['pid']==identity['supervisor'] and deployment['start']==identity['supervisor_start']
assert not Path('/proc',str(deployment['pid'])).exists()
assert identity['complete'] and len(identity['runs'])==8 and (base/('complete-'+phase+'.txt')).read_text().strip()=='0'
assert all(r['code']==0 and not Path('/proc',str(r['pid'])).exists() for r in identity['runs'])
groups={r['pid'] for r in identity['runs']}|{deployment['pid']}
for p in Path('/proc').iterdir():
    if not p.name.isdigit():continue
    try:
        line=(p/'stat').read_text();fields=line[line.rfind(')')+2:].split()
        assert int(fields[2]) not in groups or fields[0]=='Z',line
    except (FileNotFoundError,ProcessLookupError):pass
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
pins={str(p.relative_to(base)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(base.rglob('*')) if p.is_file()}
receipt=dict(schema=1,phase=phase,collected_at=time.time(),supervisor=dict(pid=identity['supervisor'],start=identity['supervisor_start']),
    terminal_workers=[dict(pid=r['pid'],start=r['start_identity'],code=r['code']) for r in identity['runs']],files=pins,
    checkout=subprocess.check_output(['git','-C','/home/vermorel/Onnx','rev-parse','HEAD'],text=True).strip())
path=base/('collection-'+phase+'.json')
with path.open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
with tarfile.open(archive,'x:gz') as t:
    for p in sorted(base.rglob('*')):
        if p.is_file():t.add(p,arcname=p.relative_to(base).as_posix())
print(json.dumps(dict(archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),collection_sha256=sha(path))))
