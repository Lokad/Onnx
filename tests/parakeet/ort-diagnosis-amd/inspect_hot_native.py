"""Inspect sampled addresses in the exact installed ELF; no inference or rebuild."""
import json
from pathlib import Path
from run import ROOT, PRELUDE, pin, read, write, ssh


def main():
    base = ROOT/'artifacts/parakeet-ort-native-samples-20260924'
    result = read(base/'collected/requests/result.json')
    libraries = {p:v for p,v in result['numeric_libraries'].items() if 'onnxruntime_pybind' in p}
    assert len(libraries) == 1
    path, identity = next(iter(libraries.items()))
    state = read(base/'terminal.json')['state']; assert state['complete'] and state['code'] == 0
    ranges = [(0x1246800,0x124a400),(0x11cae00,0x11cbc00)]
    value = ssh(PRELUDE+f'''
import re
sys.path.insert(0,str(base))
from remote import live,pin
assert not live({state['supervisor']!r}) and all(not live(dict(pid=int(p),birth=b)) for p,b in {state['members']!r}.items())
path={path!r};assert pin(path)=={identity!r}
def command(args):
 p=subprocess.run(args,text=True,capture_output=True,timeout=30)
 return dict(command=args,code=p.returncode,stdout=p.stdout,stderr=p.stderr)
assembly=[command(['/usr/bin/objdump','-d','-M','intel','--start-address='+hex(a),'--stop-address='+hex(b),path]) for a,b in {ranges!r}]
frames=command(['/usr/bin/readelf','--debug-dump=frames',path])
assert frames['code']==0
entries=[]
for line in frames['stdout'].splitlines():
 m=re.search(r'FDE.*pc=([0-9a-f]+)[.][.]([0-9a-f]+)',line)
 if m and any(int(m[1],16)<b and int(m[2],16)>a for a,b in {ranges!r}):entries.append(line)
stats=command(['/usr/bin/perf','report','--stdio','--header-only','--stats','-i','/dev/shm/lokad-parakeet-ort-native-samples-20260924/perf.data'])
print(json.dumps(dict(binary=pin(path),path=path,ranges={ranges!r},assembly=assembly,frame_entries=entries,perf_stats=stats,inference_calls=0)))
''')
    write(base/'native-inspection.json',value)
    for i, part in enumerate(value['assembly']):
        assert part['code'] == 0
        (base/f'assembly-{i}.txt').write_text(part['stdout'],encoding='utf8')
    print(json.dumps(dict(binary=identity,frame_entries=value['frame_entries'],stats=value['perf_stats']['stdout'][:10000],
                         stats_stderr=value['perf_stats']['stderr'])))


if __name__ == '__main__':
    main()
