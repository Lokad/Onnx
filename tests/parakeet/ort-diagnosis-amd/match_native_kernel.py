"""Match one pinned MLAS assembly routine to the sampled installed binary bytes."""
import base64
import json
from pathlib import Path
import subprocess
from run import ROOT, PRELUDE, pin, read, write, ssh


def main():
    out = ROOT/'artifacts/parakeet-ort-native-kernel-match-20260924'
    assert not out.exists(); out.mkdir()
    revision = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'
    files = {}
    for name in ['SgemmKernelAvx512F.S','asmmacro.h','SgemmKernelCommon.h',
                 'FgemmKernelCommon.h','FgemmKernelAvx512FCommon.h']:
        p = subprocess.run(['git','-c','gc.auto=0','-C',str(ROOT/'external/onnxruntime'),'show',
                            revision+':onnxruntime/core/mlas/lib/x86_64/'+name],capture_output=True,check=True)
        (out/name).write_bytes(p.stdout); files[name] = pin(out/name)
    samples = ROOT/'artifacts/parakeet-ort-native-samples-20260924'
    inspection = read(samples/'native-inspection.json')
    payload = {name:base64.b64encode((out/name).read_bytes()).decode('ascii') for name in files}
    spec = dict(revision=revision,files=files,binary=inspection['binary'],path=inspection['path'],
                inspected_start=0x1247970,source=pin(__file__),inference_calls=0,product_rebuild=False)
    write(out/'prepared.json',spec)
    value = ssh(PRELUDE+f'''
import base64,resource
from pathlib import Path
os.sched_setaffinity(0,{{2}})
folder=Path('/dev/shm/lokad-parakeet-ort-native-kernel-match-20260924')
assert not folder.exists();folder.mkdir()
assert psutil.virtual_memory().available>=2*1024**3 and psutil.disk_usage(folder).free>=1024**3
for name,data in {payload!r}.items():(folder/name).write_bytes(base64.b64decode(data))
def digest(path):
 with Path(path).open('rb') as stream:return dict(bytes=Path(path).stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())
for name,wanted in {files!r}.items():assert digest(folder/name)==wanted
assert digest({inspection['path']!r})=={inspection['binary']!r}
def limit():
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
 resource.setrlimit(resource.RLIMIT_CPU,(20,20))
commands=[]
for args in [
 ['/usr/bin/gcc','-c','-x','assembler-with-cpp','-I',str(folder),str(folder/'SgemmKernelAvx512F.S'),'-o',str(folder/'kernel.o')],
 ['/usr/bin/objcopy','-O','binary','--only-section=.text',str(folder/'kernel.o'),str(folder/'kernel.text')],
 ['/usr/bin/readelf','--wide','--symbols','--relocs',str(folder/'kernel.o')]]:
 p=subprocess.run(args,text=True,capture_output=True,timeout=30,preexec_fn=limit)
 commands.append(dict(command=args,code=p.returncode,stdout=p.stdout,stderr=p.stderr))
 if p.returncode:break
value=dict(commands=commands,passed=False)
if all(c['code']==0 for c in commands):
 candidate=(folder/'kernel.text').read_bytes()
 with Path({inspection['path']!r}).open('rb') as stream:
  stream.seek(0x1247970);original=stream.read(len(candidate))
 assert len(candidate)<65536
 value.update(passed=candidate==original,bytes=len(candidate),start=0x1247970,end=0x1247970+len(candidate),
  candidate_sha256=hashlib.sha256(candidate).hexdigest(),original_sha256=hashlib.sha256(original).hexdigest(),
  candidate_base64=base64.b64encode(candidate).decode('ascii'),original_base64=base64.b64encode(original).decode('ascii'))
print(json.dumps(value))
''')
    for name in ['candidate','original']:
        key=name+'_base64'
        if key in value:
            (out/(name+'.bin')).write_bytes(base64.b64decode(value.pop(key)))
    write(out/'analysis.json',value)
    write(out/'closed.json',dict(passed=value['passed'],prepared=pin(out/'prepared.json'),analysis=pin(out/'analysis.json'),
                                inference_calls=0,product_rebuild=False))
    print(json.dumps({k:v for k,v in value.items() if k!='commands'}))
    if not value['passed']:
        print(json.dumps(value['commands']))


if __name__ == '__main__':
    main()
