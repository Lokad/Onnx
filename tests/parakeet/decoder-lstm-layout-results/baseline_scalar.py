"""One diagnostic control using the qualified backend binary; no rebuild or retry."""
import base64
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'artifacts/parakeet-decoder-lstm-layout-scalar-baseline-20260927'
V3 = ROOT/'artifacts/parakeet-decoder-lstm-layout-contracts-v3-amd-20260927'
QUALIFIED = ROOT/'artifacts/parakeet-decoder-packed-row-root-amd-20260927'
TOOLS = ROOT/'tests/parakeet/decoder-lstm-layout'
sys.path.insert(0, str(TOOLS))
from protocol import pin, read, save
import run

REMOTE = '/dev/shm/lokad-lstmlayout-baseline-20260927'
VM_ROOT = '/dev/shm/lokad-parakeet-decoder-packed-row-root-20260927'

PROBE = r'''import json,os,subprocess,sys,time
from pathlib import Path
import xml.etree.ElementTree as ET
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(sys.argv[1]);spec=json.loads((base/'prospective.json').read_text())
out=base/'baseline-scalar';runtime=base/'runtime'
env=dict(os.environ,DOTNET_EnableHWIntrinsic='0',DOTNET_CLI_HOME=str(base/'cli-home'),
 DOTNET_SKIP_FIRST_TIME_EXPERIENCE='1',DOTNET_CLI_TELEMETRY_OPTOUT='1')
command=['/home/vermorel/.dotnet/dotnet','vstest',str(runtime/'Lokad.Onnx.Backend.Tests.dll'),
 '--TestCaseFilter:FullyQualifiedName~Lstm|FullyQualifiedName~LSTM',
 '--Logger:trx;LogFileName=contracts.trx','--ResultsDirectory:'+str(out)]
child=subprocess.Popen(command,env=env)
observed={}
while child.poll() is None:
 try:
  for process in [psutil.Process(child.pid),*psutil.Process(child.pid).children(recursive=True)]:
   try:
    for entry in process.memory_maps():
     if Path(entry.path).name in ['Lokad.Onnx.dll','Lokad.Onnx.Backend.Tests.dll']:
      assert Path(entry.path).parent==runtime,entry.path
      observed.setdefault(str(process.pid),dict(birth=process.create_time(),modules=[]))
      if entry.path not in observed[str(process.pid)]['modules']:observed[str(process.pid)]['modules'].append(entry.path)
   except (psutil.NoSuchProcess,psutil.ZombieProcess):pass
 except psutil.NoSuchProcess:pass
 time.sleep(.025)
code=child.wait();assert code==1,code
ns={'t':'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
rows=ET.parse(out/'contracts.trx').getroot().findall('.//t:UnitTestResult',ns)
assert len(rows)==172 and len({r.attrib['testName'] for r in rows})==172
assert {r.attrib['testName'] for r in rows}==set(spec['expected_census'])
failed={r.attrib['testName']:r.find('.//t:Message',ns).text.strip() for r in rows if r.attrib['outcome']!='Passed'}
assert failed==spec['expected_rejections'] and len(failed)==18
assert all(r.attrib['outcome'] in ['Passed','Failed'] for r in rows)
assert any(set(v['modules'])=={str(runtime/'Lokad.Onnx.dll'),str(runtime/'Lokad.Onnx.Backend.Tests.dll')} for v in observed.values())
with (out/'result.json').open('x') as f:json.dump(dict(passed=True,diagnostic_only=True,
 pid=os.getpid(),child_pid=child.pid,test_exit_code=code,passed_cases=154,unsupported_cases=18,
 expected_rejections=failed,loaded=observed,command=command,hardware_override={'DOTNET_EnableHWIntrinsic':'0'}),f,indent=2)
'''

DRIVER = r'''from pathlib import Path
import base64,hashlib,json,os,sys
sys.path.insert(0,'/home/vermorel/Onnx/artifacts/asr-multilingual-amd-20260920/python')
import psutil
base=Path(REMOTE)
os.sched_setaffinity(0,{0})
assert not base.exists() and psutil.boot_time()==1789634288.0
own=psutil.Process();ancestors={own.pid,*[p.pid for p in own.parents()]}
for p in psutil.process_iter(['pid','name','cmdline']):
 if p.pid in ancestors:continue
 assert p.info['name'] not in ['dotnet','perf']
 assert not (p.info['name'].startswith('python') and '/dev/shm/lokad-' in ' '.join(p.info['cmdline'] or []))
assert psutil.virtual_memory().available>=4*1024**3 and psutil.disk_usage('/dev/shm').free>=1024**3
base.mkdir();(base/'source').mkdir();(base/'tools').mkdir();(base/'runtime').mkdir()
for name,encoded in FILES.items():
 p=base/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(encoded))
sys.path.insert(0,str(base/'tools'))
from protocol import JOBS,LIMITS,pin,read,save,verify
import remote
root=Path(VM_ROOT);prior=read(root/'payload.json');receipt=read(root/'collection.json')
assert receipt['terminal'] and receipt['code']==0 and pin(root/'collection.json')==ROOT_RECEIPT
assert not any(remote.live(i) for i in receipt['identities'])
spec=read(base/'prospective.json')
for name,wanted in spec['runtime'].items():
 source=root/'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0'/name
 assert pin(source)==wanted,name
 target=base/'runtime'/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(source,target)
external=dict(prior['external'])
for name,wanted in external.items():assert pin(name)==wanted,name
payload=dict(passed=True,jobs=JOBS,limits=LIMITS,boot_time=1789634288.0,
 interpreter=prior['interpreter'],previous_owner=receipt['identities'][0],external=external,
 files={p.relative_to(base).as_posix():pin(p) for p in base.rglob('*') if p.is_file()})
save(base/'payload.json',payload);verify(base)
save(base/'deployment.json',dict(pid=own.pid,birth=own.create_time()))
remote.command_for=lambda name,spec:([sys.executable,'-B',base/'probe.py',base],False,2)
def after(name,spec,row):
 result=read(base/name/'result.json')
 assert result['passed'] and result['pid']==row['processes']['worker']['pid']
 for pid,entry in result['loaded'].items():assert row['members'][pid]==entry['birth']
remote.after=after
code=remote.main()
print(json.dumps(dict(code=code,deployment=read(base/'deployment.json'),
 payload_base64=base64.b64encode((base/'payload.json').read_bytes()).decode())))
'''


def main():
    assert not OUT.exists()
    assert pin(V3/'failed.json')['sha256'] == '59f6f738dcd71beec865e46c511a9bd151207a26d2901c259f015b387a27f63f'
    for name, wanted in read(V3/'failed.json')['files'].items(): assert pin(V3/name) == wanted, name
    assert pin(QUALIFIED/'closed.json')['sha256'] == 'd0a78cdd3106d6a72a41303f879bbb2f9ea3bd6a298d778333015f38ccdac246'
    prefix = 'source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0/'
    runtime = {n.removeprefix(prefix): wanted for n, wanted in read(QUALIFIED/'collected/built.json')['files'].items() if n.startswith(prefix)}
    assert len(runtime) == 116
    stage = read(V3/'bundle/stage.json')
    ns = {'t': 'http://microsoft.com/schemas/VisualStudio/TeamTest/2010'}
    rows = ET.parse(V3/'collected/contracts-scalar/contracts.trx').getroot().findall('.//t:UnitTestResult', ns)
    rejected = {r.attrib['testName']: r.find('.//t:Message', ns).text.strip() for r in rows if r.attrib['outcome'] != 'Passed'}
    assert len(rejected) == 18 and len(stage['expected_census']) == 172
    assert all('LstmReferenceTests.MultipleBatchesDirectionsActivationsAndStorageMatchOrt' in n and 'mode: 2)' in n for n in rejected)
    assert set(rejected.values()) == {'System.InvalidOperationException : Tensor intrinsics were explicitly requested but x86 FMA is not supported on this machine.'}
    prospective = dict(diagnostic_only=True, no_build=True, performance_measured=False,
        runtime=runtime, expected_census=stage['expected_census'], expected_rejections=rejected,
        candidate_failure=pin(V3/'failed.json'), qualified_root=pin(QUALIFIED/'closed.json'))
    protocol = (V3/'bundle/tools/protocol.py').read_text()
    start = protocol.index('JOBS = '); end = protocol.index('\nGIB = ', start)
    protocol = protocol[:start] + "JOBS = ['baseline-scalar']" + protocol[end:]
    files = {'tools/protocol.py': protocol.encode(), 'probe.py': PROBE.encode(),
        'prospective.json': json.dumps(prospective, indent=2).encode(),
        'tools/protocol_base.py': (V3/'bundle/tools/protocol_base.py').read_bytes(),
        'tools/remote.py': (V3/'bundle/tools/remote_base.py').read_bytes()}
    OUT.mkdir()
    for name, data in files.items():
        p=OUT/'bundle'/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    driver = 'REMOTE='+repr(REMOTE)+'\nVM_ROOT='+repr(VM_ROOT)+'\nROOT_RECEIPT='+repr(pin(QUALIFIED/'collected/collection.json'))+'\nFILES='+repr({n:base64.b64encode(b).decode() for n,b in files.items()})+'\n'+DRIVER
    (OUT/'driver.py').write_text(driver, encoding='utf8', newline='\n')
    save(OUT/'prepared.json', dict(passed=True, source=pin(Path(__file__)), files={p.relative_to(OUT).as_posix():pin(p) for p in (OUT/'bundle').rglob('*') if p.is_file()}, driver=pin(OUT/'driver.py')))
    response = run.ssh(driver, 180)
    (OUT/'driver.stdout').write_text(response, encoding='utf8')
    result = json.loads(response.splitlines()[-1])
    (OUT/'payload.json').write_bytes(base64.b64decode(result.pop('payload_base64')))
    save(OUT/'deployment.json', result['deployment']); save(OUT/'execution.json', result)
    # The established collector verifies terminal PID/birth identities and every byte.
    transport = run.transport
    transport.BASE = OUT
    transport.PRELUDE = transport.PRELUDE.replace(run.REMOTE, REMOTE)
    def frozen():
        value=read(OUT/'prepared.json');assert value['source']==pin(Path(__file__)) and value['driver']==pin(OUT/'driver.py')
        for name,wanted in value['files'].items():assert pin(OUT/name)==wanted,name
        return value
    transport.prepared = frozen
    transport.collect()
    assert result['code'] == 0
    print(json.dumps(dict(diagnostic_passed=True, result=read(OUT/'collected/baseline-scalar/result.json'))))


if __name__ == '__main__': main()
