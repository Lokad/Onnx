"""Retain the rejected forced-interval assumption without rerunning its proof."""
import json
from pathlib import Path
from run import ROOT, PRELUDE, transport_ssh, pin, read, save

BASE = ROOT/'artifacts/e5-frequency-proof-v2-amd-20260925'
TOOLS = ROOT/'tests/benchmarks/e5-frequency-probe-v2'


def main():
    assert not (BASE/'closed.json').exists()
    spec=read(BASE/'prepared.json'); receipt=read(BASE/'collected.json')
    for name,wanted in spec['files'].items(): assert pin(TOOLS/name)==wanted,name
    for name,wanted in receipt['files'].items(): assert pin(BASE/'collected'/name)==wanted,name
    state=read(BASE/'collected/state.json')
    assert receipt['terminal'] and state['complete'] and state['terminal'] and state['code']==1
    assert state['owner']==receipt['owner']==read(BASE/'deployment.json')
    assert state['exitcodes']==dict(frequency=0,workload=0) and 'workload_command' in state
    assert 'startswith(received)' in state['error']
    assert len(state['runtime_clock']['brackets'])==1000
    helper=read(BASE/'collected/counters/ready.json')
    terminal=json.loads(transport_ssh(PRELUDE+f'''
idle()
assert not live({state['owner']!r})
assert all(not live(dict(pid=int(pid),birth=birth)) for pid,birth in {state['identities']!r}.items())
assert not psutil.pid_exists({helper['pid']!r})
print(json.dumps(dict(idle=True,original_owner_terminal=True,helper_pid_absent=True,checked=time.time())))
'''))
    analysis=dict(passed=False,terminal=terminal,inference_calls=0,
        reason='The acknowledgment reader consumes four bytes but perf writes five including its NUL terminator; the next acknowledgment is rejected. No model executes.',
        runtime_timestamp_brackets_passed=1000,source=pin(Path(__file__)),error=state['error'])
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=False,terminal=True,analysis=pin(BASE/'analysis.json'),files=files,inference_calls=0))
    print(json.dumps(dict(closure=pin(BASE/'closed.json'),passed=False,inference_calls=0,terminal=terminal)))


if __name__=='__main__':main()
