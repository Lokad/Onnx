"""One bounded assembly identification; reuse original native samples, no inference."""
import base64
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/ort-diagnosis-amd'))
from run import PRELUDE, pin, read, ssh, write

BASE = ROOT/'artifacts/parakeet-ort-depthwise-kernels-20260925'
REMOTE = '/dev/shm/lokad-parakeet-ort-depthwise-kernels-20260925'
REV = '2e2543fbe9fae542f921d47a72d21d5a4ef0b710'


def main():
    assert not BASE.exists()
    counts = ROOT/'artifacts/parakeet-depthwise-route-amd-20260925'
    count_closed = read(counts/'closed.json')
    assert count_closed['passed'] and count_closed['analysis'] == pin(counts/'analysis.json')
    samples = ROOT/'artifacts/parakeet-ort-native-samples-20260924'
    closed = read(samples/'closed.json'); analysis = read(samples/'analysis.json')
    inspection = read(samples/'native-inspection.json')
    assert closed['passed'] and closed['analysis'] == pin(samples/'analysis.json')
    assert closed['inspection'] == pin(samples/'native-inspection.json')
    assert analysis['loaded_binary'] == inspection['binary']
    BASE.mkdir(); files = {}
    for name in ['SconvKernelAvx512F.S', 'SconvKernelCommon.h', 'SgemmKernelM1Avx.S', 'asmmacro.h', '../platform.cpp']:
        path = 'onnxruntime/core/mlas/lib/'+('platform.cpp' if name == '../platform.cpp' else 'x86_64/'+name)
        content = subprocess.check_output(['git', '-c', 'gc.auto=0', '-C', str(ROOT/'external/onnxruntime'),
                                          'show', REV+':'+path], env=dict(os.environ, GIT_NO_LAZY_FETCH='1'), timeout=15)
        local = BASE/Path(name).name; local.write_bytes(content); files[local.name] = pin(local)
    assert b'MlasMaskMoveAvx[8], 32) = { 0, 1, 2, 3, 4, 5, 6, 7 }' in (BASE/'platform.cpp').read_bytes()
    (BASE/'match.py').write_bytes(Path(__file__).with_name('match.py').read_bytes()); files['match.py'] = pin(BASE/'match.py')
    spec = dict(revision=REV, sources=files, script=pin(__file__), binary=inspection['binary'],
                samples=pin(samples/'closed.json'), analysis=pin(samples/'analysis.json'),
                managed_count_closure=pin(counts/'closed.json'), inference_calls=0, product_rebuild=False)
    write(BASE/'prepared.json', spec)
    payload = {n:base64.b64encode((BASE/n).read_bytes()).decode() for n in files}
    script = PRELUDE+f'''
import base64,resource,traceback
sys.path.insert(0,'/dev/shm/lokad-parakeet-depthwise-route-20260925')
from common import idle,live
assert psutil.boot_time()==1789634288.0
assert all(not live(i) for i in {count_closed['terminal_owners']!r})
idle();os.sched_setaffinity(0,{{2}})
assert psutil.virtual_memory().available>=2*1024**3 and psutil.disk_usage('/dev/shm').free>=1024**3
folder=Path({REMOTE!r});assert not folder.exists();folder.mkdir()
for name,data in {payload!r}.items():(folder/name).write_bytes(base64.b64decode(data))
def digest(p):
 p=Path(p)
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
for name,wanted in {files!r}.items():assert digest(folder/name)==wanted
assert digest({inspection['path']!r})=={inspection['binary']!r}
def limit():
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(20,20))
result=dict(passed=False,commands=[],matches=[],inference_calls=0,product_rebuild=False)
try:
 sys.path.insert(0,str(folder));from match import match_object
 binary=Path({inspection['path']!r}).read_bytes()
 for source,target in [('SconvKernelAvx512F.S','MlasConvDepthwiseFloatKernelAvx512F'),('SgemmKernelM1Avx.S','MlasSgemmKernelM1Avx')]:
  output=folder/(source+'.o')
  args=['/usr/bin/gcc','-c','-x','assembler-with-cpp','-I',str(folder),str(folder/source),'-o',str(output)]
  run=subprocess.run(args,text=True,capture_output=True,timeout=30,preexec_fn=limit)
  result['commands'].append(dict(command=args,code=run.returncode,stdout=run.stdout,stderr=run.stderr))
  assert run.returncode==0 and not run.stderr
  match=match_object(output.read_bytes(),binary,target)
  match['object']=digest(output);match['object_base64']=base64.b64encode(output.read_bytes()).decode()
  match['native_base64']=base64.b64encode(binary[match['start']:match['end']]).decode()
  result['matches'].append(match)
 assert digest({inspection['path']!r})=={inspection['binary']!r}
 result['passed']=True
except Exception:result['error']=traceback.format_exc()
result['output_bytes']=sum(p.stat().st_size for p in folder.iterdir() if p.is_file())
assert result['output_bytes']<1024**2
print(json.dumps(result))
'''
    (BASE/'remote.py').write_text(script, encoding='utf8')
    result = ssh(script)
    for match in result['matches']:
        for key in ['object', 'native']:
            (BASE/(match['target']+'.'+key)).write_bytes(base64.b64decode(match.pop(key+'_base64')))
        hits = [r for r in analysis['instruction_addresses'] if match['start'] <= int(r['offset'], 16) < match['end']]
        weight = sum(r['period_ns'] for r in hits)
        match.update(sampled_instruction_addresses=hits, measured_sample_period_ns=weight,
                     estimated_seconds_per_corpus=weight/3e9, observed=bool(hits),
                     share=weight/analysis['measured_period_ns'])
    write(BASE/'analysis.json', result)
    write(BASE/'closed.json', dict(passed=result['passed'], prepared=pin(BASE/'prepared.json'), analysis=pin(BASE/'analysis.json'),
                                  files={p.name:pin(p) for p in BASE.iterdir() if p.is_file()}))
    print(json.dumps(dict(passed=result['passed'], matches=[{k:v for k,v in m.items() if k not in ['symbols','sampled_instruction_addresses']} for m in result['matches']], error=result.get('error'))))


if __name__ == '__main__': main()
