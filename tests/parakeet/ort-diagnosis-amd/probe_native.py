"""Read native binary metadata and test perf access after the diagnostic is closed."""
import json
from pathlib import Path
from run import BASE, ROOT, PRELUDE, ssh, read, pin, write


def main():
    proof = read(BASE/'closed.json'); assert proof['passed']
    assert proof['analysis'] == pin(BASE/'analysis.json')
    original = read(BASE/'collected/control/requests/result.json')
    libraries = {path: value for path, value in original['numeric_libraries'].items() if 'onnxruntime' in path}
    assert libraries
    out = ROOT/'artifacts/parakeet-ort-native-capabilities-20260924'
    assert not out.exists(); out.mkdir()
    value = ssh(PRELUDE+f'''
sys.path.insert(0,str(base))
from remote import live,pin,read
assert read(base/'state.json')['complete'] and read(base/'state.json')['code']==0
assert all(not live(i) for i in {proof['terminal_owners']!r})
libraries={libraries!r}
for path,wanted in libraries.items():assert pin(path)==wanted
def command(args):
 started=time.monotonic()
 p=subprocess.run(args,text=True,capture_output=True,timeout=30)
 return dict(command=args,code=p.returncode,stdout=p.stdout,stderr=p.stderr,seconds=time.monotonic()-started)
commands=[command(['/usr/bin/perf','--version']),
 command(['/usr/bin/perf','stat','-e','task-clock','--','/usr/bin/true']),
 command(['sudo','-n','/usr/bin/perf','stat','-e','task-clock','--','/usr/bin/true'])]
binary_metadata={{}}
for path in libraries:
 binary_metadata[path]=[command(['/usr/bin/readelf','--wide','--sections','--notes',path]),
  command(['/usr/bin/nm','--dynamic','--defined-only',path])]
print(json.dumps(dict(libraries=libraries,commands=commands,binary_metadata=binary_metadata,
 kernel=os.uname().release,inference_calls=0,system_settings_changed=False)))
''')
    write(out/'observations.json', value)
    write(out/'closed.json', dict(passed=True, previous=pin(BASE/'closed.json'),
        source=pin(__file__), observations=pin(out/'observations.json'), inference_calls=0))
    print(json.dumps(dict(commands=[{k:c[k] for k in ['command','code','stdout','stderr']} for c in value['commands']],
        libraries=list(libraries), observation=pin(out/'observations.json'))))


if __name__ == '__main__':
    main()
