"""Remove only verified idle VM duplicates of the closed eight-process observer."""
import json
from pathlib import Path
import retire_closed_output_duplicates as common

ROOT=common.ROOT
SOURCE=ROOT/'artifacts/e5-repeatability-diagnostic-amd-20260925'
OUT=ROOT/'artifacts/e5-repeatability-closed-trace-retention-20260925'
REMOTE='/dev/shm/lokad-e5-repeatability-diagnostic-20260925'
pin,read,save,ssh=common.pin,common.read,common.save,common.ssh


def main():
    assert not OUT.exists()
    assert pin(SOURCE/'closed.json')['sha256']=='fe8ec805d18864a3a8d8b5a052543a3eef466535adc2769efc8c6c6747338a7b'
    proof=read(SOURCE/'closed.json');assert proof['passed'] and proof['diagnostic_only']
    for name,wanted in proof['files'].items():assert pin(SOURCE/name)==wanted,name
    receipt=read(SOURCE/'collected/collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    names=[r+'-capture/capture.nettrace' for r in 'abcdefgh']
    names += [r+'-export/events/events.jsonl.gz' for r in 'abcdefgh']
    names += ['exporter-roundtrip/events/events.jsonl.gz']
    targets={n:pin(SOURCE/'collected'/n) for n in names}
    assert all(receipt['files'][n]==v for n,v in targets.items())
    script=common.COMMON+f'''
root=Path({REMOTE!r});targets={targets!r}
assert root.resolve()==root and root.parent==Path('/dev/shm')
assert pin(root/'collection.json')=={pin(SOURCE/'collected/collection.json')!r}
assert not any(live(i) for i in {receipt['identities']!r})
for name,wanted in targets.items():
 path=root/name
 assert path.resolve()==path and path.is_relative_to(root) and not path.is_symlink()
 assert path.stat().st_nlink==1 and str(path) not in protected and pin(path)==wanted
physical=sum((root/name).stat().st_blocks*512 for name in targets)
idle()
'''
    snapshot=json.loads(ssh(script+'''
print(json.dumps(dict(passed=True,manifests=manifests,physical_bytes=physical,
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''',300))
    OUT.mkdir();save(OUT/'prospective.json',dict(**snapshot,targets=targets,source=pin(SOURCE/'closed.json'),
        generator=pin(Path(__file__)),common=pin(Path(common.__file__))))
    result=json.loads(ssh(script+f'''
assert manifests=={snapshot['manifests']!r} and physical=={snapshot['physical_bytes']!r}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for name in targets:(root/name).unlink()
assert all(not (root/name).exists() for name in targets)
idle()
print(json.dumps(dict(passed=True,files=len(targets),physical_bytes=physical,
 logical_bytes=sum(v['bytes'] for v in targets.values()),before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''',300))
    for name,wanted in targets.items():assert pin(SOURCE/'collected'/name)==wanted,name
    save(OUT/'closed.json',dict(**result,prospective=pin(OUT/'prospective.json'),all_local_originals_retained=True))
    print(json.dumps(result))


if __name__=='__main__':main()
