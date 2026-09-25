"""Preserve the unsupported-relocation failure; reuse its object and finish once."""
import base64
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/ort-diagnosis-amd'))
from run import PRELUDE, pin, read, ssh, write

BASE = ROOT/'artifacts/parakeet-ort-depthwise-kernels-20260925'
REMOTE = '/dev/shm/lokad-parakeet-ort-depthwise-kernels-20260925'


def main():
    assert not (BASE/'corrected-closed.json').exists()
    failed = read(BASE/'closed.json'); assert not failed['passed']
    for name, wanted in failed['files'].items(): assert pin(BASE/name) == wanted
    original = read(BASE/'prepared.json')
    samples = ROOT/'artifacts/parakeet-ort-native-samples-20260924'
    analysis = read(samples/'analysis.json'); inspection = read(samples/'native-inspection.json')
    assert pin(samples/'analysis.json') == original['analysis']
    counts = read(ROOT/'artifacts/parakeet-depthwise-route-amd-20260925/closed.json')
    source = Path(__file__).with_name('match_internal.py')
    (BASE/source.name).write_bytes(source.read_bytes())
    correction = dict(failed_closure=pin(BASE/'closed.json'),
                      retained_object=pin(BASE/'retained-conv.object'),
                      relocation_inspection=pin(BASE/'relocation-inspection.json'),
                      matcher=pin(source), runner=pin(__file__), inference_calls=0,
                      convolution_assembly_repeated=False, first_single_row_assembly=True)
    write(BASE/'correction.json', correction)
    encoded = base64.b64encode(source.read_bytes()).decode()
    script = PRELUDE+f'''
import base64,resource,traceback
sys.path.insert(0,'/dev/shm/lokad-parakeet-depthwise-route-20260925')
from common import idle,live
assert all(not live(i) for i in {counts['terminal_owners']!r});idle()
os.sched_setaffinity(0,{{2}})
assert psutil.virtual_memory().available>=2*1024**3 and psutil.disk_usage('/dev/shm').free>=1024**3
folder=Path({REMOTE!r})
def digest(p):
 p=Path(p)
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
for name,wanted in {original['sources']!r}.items():assert digest(folder/name)==wanted
assert digest(folder/'SconvKernelAvx512F.S.o')=={correction['retained_object']!r}
assert digest({inspection['path']!r})=={original['binary']!r}
with (folder/'match_internal.py').open('xb') as f:f.write(base64.b64decode({encoded!r}))
def limit():
 resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3));resource.setrlimit(resource.RLIMIT_CPU,(20,20))
result=dict(passed=False,commands=[],matches=[],inference_calls=0)
try:
 sys.path.insert(0,str(folder));from match_internal import match_object
 output=folder/'SgemmKernelM1Avx.S.o';assert not output.exists()
 args=['/usr/bin/gcc','-c','-x','assembler-with-cpp','-I',str(folder),str(folder/'SgemmKernelM1Avx.S'),'-o',str(output)]
 run=subprocess.run(args,text=True,capture_output=True,timeout=30,preexec_fn=limit)
 result['commands'].append(dict(command=args,code=run.returncode,stdout=run.stdout,stderr=run.stderr))
 assert run.returncode==0 and not run.stderr
 binary=Path({inspection['path']!r}).read_bytes()
 for name,target in [('SconvKernelAvx512F.S','MlasConvDepthwiseFloatKernelAvx512F'),('SgemmKernelM1Avx.S','MlasSgemmKernelM1Avx')]:
  obj=folder/(name+'.o');match=match_object(obj.read_bytes(),binary,target)
  match['object']=digest(obj);match['object_base64']=base64.b64encode(obj.read_bytes()).decode()
  match['native_base64']=base64.b64encode(binary[match['start']:match['end']]).decode()
  result['matches'].append(match)
 result['passed']=True
except Exception:result['error']=traceback.format_exc()
assert digest({inspection['path']!r})=={original['binary']!r}
result['output_bytes']=sum(p.stat().st_size for p in folder.iterdir() if p.is_file());assert result['output_bytes']<1024**2
print(json.dumps(result))
'''
    (BASE/'corrected-remote.py').write_text(script, encoding='utf8')
    result = ssh(script)
    for match in result['matches']:
        for key in ['object', 'native']:
            with (BASE/(match['target']+'.'+key)).open('xb') as f:
                f.write(base64.b64decode(match.pop(key+'_base64')))
        hits = [r for r in analysis['instruction_addresses'] if match['start'] <= int(r['offset'], 16) < match['end']]
        weight = sum(r['period_ns'] for r in hits)
        match.update(sampled_instruction_addresses=hits, measured_sample_period_ns=weight,
                     estimated_seconds_per_corpus=weight/3e9, observed=bool(hits),
                     share=weight/analysis['measured_period_ns'])
    write(BASE/'corrected-analysis.json', result)
    write(BASE/'corrected-closed.json', dict(passed=result['passed'], correction=pin(BASE/'correction.json'),
        analysis=pin(BASE/'corrected-analysis.json'), files={p.name:pin(p) for p in BASE.iterdir() if p.is_file()}))
    print(json.dumps(dict(passed=result['passed'], matches=[{k:v for k,v in m.items() if k not in
        ['symbols','complete_text','sampled_instruction_addresses']} for m in result['matches']], error=result.get('error'))))


if __name__ == '__main__': main()
