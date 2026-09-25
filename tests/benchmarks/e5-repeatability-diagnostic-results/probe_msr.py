"""Check the frequency counters actually advertised by this VM, without inference."""
import json
from pathlib import Path
from probe_counters import ROOT, PYANNOTE, PRELUDE, ssh, pin, read, save

BASE = ROOT/'artifacts/e5-frequency-counter-capabilities-20260925'
PREVIOUS = ROOT/'artifacts/e5-hardware-counter-capabilities-20260925'


def main():
    assert not BASE.exists()
    proof = read(PREVIOUS/'closed.json')
    assert proof['passed'] and proof['observations'] == pin(PREVIOUS/'observations.json')
    observations = read(PREVIOUS/'observations.json')
    assert observations['sources']['msr']['events'] == ['aperf', 'mperf', 'tsc']
    receipt = read(PYANNOTE/'collected/collection.json')
    assert receipt['terminal'] and receipt['code'] == 0
    script = PRELUDE + f'''
from remote import idle,live
idle()
assert psutil.boot_time()==1789634288.0
assert not any(live(i) for i in {receipt['identities']!r})
args=['sudo','-n','/usr/bin/perf','stat','-a','-C','2','-x',';',
 '-e','msr/aperf/,msr/mperf/,msr/tsc/','--','/usr/bin/taskset','-c','2',
 sys.executable,'-B','-c','value=0\\nfor i in range(1000000): value=(value+i)%104729\\nassert value>=0']
started=time.monotonic()
result=subprocess.run(args,text=True,capture_output=True,timeout=15)
idle()
print(json.dumps(dict(command=args,code=result.returncode,stdout=result.stdout,
 stderr=result.stderr,seconds=time.monotonic()-started,inference_calls=0,
 system_settings_changed=False,previous_owners_terminal=True)))
'''
    result = json.loads(ssh(script, timeout=30))
    assert result['seconds'] < 15 and result['inference_calls'] == 0
    BASE.mkdir()
    save(BASE/'observations.json', result)
    save(BASE/'closed.json', dict(passed=True, source=pin(Path(__file__)),
        previous=pin(PREVIOUS/'closed.json'), observations=pin(BASE/'observations.json'),
        inference_calls=0, system_settings_changed=False))
    print(json.dumps(dict(closure=pin(BASE/'closed.json'), **result)))


if __name__ == '__main__':
    main()
