"""Deploy once, only after the preceding AMD endurance campaign is closed."""
import json,subprocess
from common import ROOT,BASE,LOCAL,SHARING,REMOTE,KEY,HOST,PRELUDE,pin,read,write,ssh


def main():
    assert not (BASE/'frozen.json').exists() and not (BASE/'deployment.json').exists()
    closed=read(SHARING/'closed.json');assert closed['passed'] and read(SHARING/'final-verification.json')['passed']
    for name,wanted in closed['files'].items():assert pin(ROOT/name)==wanted,name
    prepared=read(BASE/'prepared.json');assert prepared['prepared'] and prepared['archive']==pin(BASE/'payload.tar.gz')
    assert prepared['local_closure']==pin(LOCAL/'local-closed.json') and prepared['inherited_frozen']==pin(SHARING/'frozen.json')
    for name,wanted in prepared['uploads'].items():assert pin(BASE/'payload'/name)==wanted,name
    script=PRELUDE+'''assert not base.exists()
assert pin(sharing/'collection.json')==%r
terminal(%r)
inherited=read(sharing/'frozen.json');assert pin(sharing/'frozen.json')==%r
for name,wanted in inherited['external'].items():assert pin(name)==wanted,name
for name,item in %r.items():assert pin(item['source'])==item['pin'],name
print(json.dumps(dict(terminal=True,available=psutil.virtual_memory().available,disk=psutil.disk_usage(str(base.parent)).free)))
'''%(pin(SHARING/'collected/collection.json'),closed['births'],prepared['inherited_frozen'],prepared['links'])
    write(BASE/'stage-check.json',json.loads(ssh(script)))
    ram='/dev/shm/whisper-memory-contracts-amd-20260921-payload.tar.gz'
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(BASE/'payload.tar.gz'),HOST+':'+ram],check=True)
    script=PRELUDE+'''assert not base.exists();archive=Path(%r);assert pin(archive)==%r
base.mkdir()
with tarfile.open(archive) as tar:tar.extractall(base,filter='data')
for name,wanted in %r.items():assert pin(base/name)==wanted,name
for name,item in %r.items():
 assert pin(item['source'])==item['pin'],name
 target=base/name;assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);os.link(item['source'],target)
inherited=read(sharing/'frozen.json')
for name,wanted in %r.items():assert inherited['external'][name]==wanted,name
frozen=dict(schema=1,scope='whisper-memory-contracts-amd',source=%r,prepared=%r,prior_closure=%r,
 limits=%r,models='/home/vermorel/Onnx/models/whisper-large-v3-turbo',managed_runtime=inherited['managed_runtime'],
 external=inherited['external'],python_paths=inherited['python_paths'],completed_requests=13,refusals=16)
frozen['files']={p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
write(base/'frozen.json',frozen);print(json.dumps(dict(frozen=pin(base/'frozen.json'),files=len(frozen['files']))))
'''%(ram,prepared['archive'],prepared['uploads'],prepared['links'],prepared['models'],prepared['source'],pin(BASE/'prepared.json'),pin(SHARING/'closed.json'),prepared['limits'])
    receipt=json.loads(ssh(script));write(BASE/'freeze-receipt.json',receipt)
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',HOST+':'+REMOTE+'/frozen.json',str(BASE/'frozen.json')],check=True)
    assert pin(BASE/'frozen.json')==receipt['frozen']
    script=PRELUDE+'''assert not (base/'run').exists() and not (base/'deployment.json').exists()
frozen=read(base/'frozen.json')
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env.update(PYTHONPATH=os.pathsep.join(frozen['python_paths']),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
with (base/'supervisor.stdout').open('x') as out,(base/'supervisor.stderr').open('x') as err:
 p=subprocess.Popen(['python3','-B',str(base/'runtime/supervise.py'),str(base)],cwd=base,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
value=dict(pid=p.pid,birth=psutil.Process(p.pid).create_time(),frozen=pin(base/'frozen.json'))
write(base/'deployment.json',value);print(json.dumps(value))
'''
    value=json.loads(ssh(script));write(BASE/'deployment.json',value);print(json.dumps(value))


if __name__=='__main__':main()
