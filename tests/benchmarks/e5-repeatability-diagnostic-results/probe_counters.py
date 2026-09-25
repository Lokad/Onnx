"""Inspect VM counter availability after all model owners are terminal."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/packed-final-row-pyannote-amd'))
from run import PRELUDE, ssh
from protocol import pin, read, save

BASE = ROOT/'artifacts/e5-hardware-counter-capabilities-20260925'
PYANNOTE = ROOT/'artifacts/parakeet-packed-final-row-pyannote-amd-20260925'
DIAGNOSTIC = ROOT/'artifacts/e5-repeatability-diagnostic-amd-20260925'


def main():
    assert not BASE.exists()
    for folder in [PYANNOTE, DIAGNOSTIC]:
        proof = read(folder/'closed.json')
        assert proof['passed'] and proof['analysis'] == pin(folder/'analysis.json')
    receipt = read(PYANNOTE/'collected/collection.json')
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    source = pin(Path(__file__))
    script = PRELUDE + f'''
from remote import idle,live
idle()
assert psutil.boot_time()==1789634288.0
assert not any(live(i) for i in {receipt['identities']!r})
def content(name):
 p=Path(name)
 return p.read_text().strip() if p.is_file() else None
sources={{}}
for folder in sorted(Path('/sys/bus/event_source/devices').iterdir()):
 sources[folder.name]=dict(type=content(str(folder/'type')),
  events=sorted(p.name for p in (folder/'events').iterdir()) if (folder/'events').is_dir() else [])
metadata={{name:content(name) for name in [
 '/proc/sys/kernel/perf_event_paranoid','/proc/sys/kernel/kptr_restrict',
 '/sys/devices/system/cpu/cpu2/cpufreq/scaling_driver',
 '/sys/devices/system/cpu/cpu2/cpufreq/scaling_governor',
 '/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq',
 '/sys/devices/system/cpu/cpu2/cpufreq/cpuinfo_max_freq',
 '/sys/devices/system/cpu/cpu2/topology/thread_siblings_list',
 '/sys/fs/cgroup/cpu.max','/sys/fs/cgroup/cpu.stat','/proc/self/cgroup']}}
commands=[]
work='value=0\\nfor i in range(100000): value=(value+i)%104729\\nassert value>=0'
for events in ['task-clock,context-switches,cpu-migrations,page-faults',
               'cycles:u,instructions:u','cache-references:u,cache-misses:u']:
 args=['sudo','-n','/usr/bin/perf','stat','-x',';','-e',events,'--',
       '/usr/bin/taskset','-c','2',sys.executable,'-B','-c',work]
 started=time.monotonic()
 result=subprocess.run(args,text=True,capture_output=True,timeout=15)
 commands.append(dict(command=args,code=result.returncode,stdout=result.stdout,
  stderr=result.stderr,seconds=time.monotonic()-started))
idle()
print(json.dumps(dict(boot=psutil.boot_time(),kernel=os.uname().release,
 sources=sources,metadata=metadata,commands=commands,inference_calls=0,
 system_settings_changed=False,previous_owners_terminal=True)))
'''
    result = json.loads(ssh(script, timeout=60))
    assert result['inference_calls'] == 0 and not result['system_settings_changed']
    assert len(result['commands']) == 3 and all(c['seconds'] < 15 for c in result['commands'])
    BASE.mkdir()
    save(BASE/'observations.json', result)
    save(BASE/'closed.json', dict(passed=True, source=source,
        pyannote=pin(PYANNOTE/'closed.json'), diagnostic=pin(DIAGNOSTIC/'closed.json'),
        observations=pin(BASE/'observations.json'), inference_calls=0,
        system_settings_changed=False))
    print(json.dumps(dict(closure=pin(BASE/'closed.json'), sources=list(result['sources']),
        commands=[{k:c[k] for k in ['code','stdout','stderr','seconds']} for c in result['commands']])))


if __name__ == '__main__':
    main()
