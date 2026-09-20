"""Audit complete process evidence without turning application mismatches into passes."""
from pathlib import Path
import argparse
import importlib.util
import json
import psutil
from common import pin, read, write
from supervise import check_sample


def module(path):
    spec=importlib.util.spec_from_file_location('retained_meeting_accounting',path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value);return value


def inspect(base,engine,frozen):
    directory=base/f'process-{engine}-run';state=read(directory/'identity.json')
    assert state['complete'] is True and 'error' not in state and state['code']==0 and read(directory/'complete.json')=={'code':0}
    assert state['mode']=='run' and state['engine']==engine and state['frozen_sha256']==pin(base/'frozen.json')['sha256']
    assert state['limits']==frozen['limits']
    assert 0<state['seconds']<state['limits']['seconds'] and state['ended']>state['started']
    samples=[json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines()]
    assert len(samples)==state['samples'] and samples
    assert all(a['seconds']<b['seconds'] for a,b in zip(samples,samples[1:]))
    for s in samples:
        check_sample(s,state['limits'])
        for m in s['members']:
            assert state['members'][str(m['pid'])]==m['birth'] and m['birth']>=state['child']['birth']
    assert state['members'][str(state['child']['pid'])]==state['child']['birth']
    peak=max(sum(m['rss'] for m in s['members']) for s in samples);assert peak==state['peak_rss']
    account=module(base/'runtime/campaign_processes.py')
    assert account.foreign_fraction(read(directory/'pre.json'),read(directory/'post.json'),state['supervisor']['pid'])==state['accounting']
    result=read(directory/'worker/result.json');assert len(result['records'])==3
    assert {p.name for p in (directory/'worker').iterdir()}=={'00.json','01.json','02.json','result.json'}
    for i,row in enumerate(result['records']):assert row==read(directory/f'worker/{i:02d}.json')
    return dict(engine=engine,seconds=state['seconds'],peak_rss=peak,minimum_available=min(s['available'] for s in samples),samples=len(samples),accounting=state['accounting'])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();base=a.artifact.resolve();frozen=read(base/'frozen.json')
    for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
    for name,wanted in frozen['native_files'].items():assert pin(Path(name))==wanted,name
    collection=read(base/'collected/collection.json');assert collection['all_owned_processes_terminal'] is True
    assert collection['frozen']==pin(base/'frozen.json') and collection['verified_reusable_files']==frozen['files']
    assert {p.relative_to(base/'collected').as_posix() for p in (base/'collected').rglob('*') if p.is_file()}==set(collection['files'])|{'collection.json'}
    for name,wanted in collection['files'].items():assert pin(base/'collected'/name)==wanted,name
    resources=[inspect(base,engine,frozen) for engine in ['native','managed']]
    native=read(base/'process-native-run/identity.json')
    births={int(pid):birth for pid,birth in native['members'].items()}
    for item in [native['supervisor'],native['child'],read(base/'deployment-native.json')]:births[item['pid']]=item['birth']
    for pid,birth in births.items():
        try:assert psutil.Process(pid).create_time()!=birth,('Owned native process still live',pid,birth)
        except psutil.NoSuchProcess:pass
    output=read(base/'process-native-run/worker/result.json')
    binary_by_name={Path(n).name:w for n,w in frozen['native_files'].items() if '/onnxruntime/capi/' in n}
    assert output['native_binaries']==binary_by_name
    result=dict(passed=True,resources=resources,native_terminal_processes=[dict(pid=p,birth=b) for p,b in births.items()],
                amd_terminal_processes=collection['terminal_processes'],frozen=pin(base/'frozen.json'),collection=pin(base/'collected/collection.json'))
    write(a.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
