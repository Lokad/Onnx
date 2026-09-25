"""Finish the retained cleanup after staging added one protected manifest."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/parakeet/feed-forward-cost-resume'))
from run import OLD, OLD_REMOTE, original, initial, paths, prepared, pin, read, write, ssh


def main():
    initial(); prepared('stages')
    out = ROOT / 'artifacts/parakeet-feed-forward-cost-original-trace-retention-20260925'
    assert not (out / 'closed.json').exists()
    proof = read(out / 'prospective.json')
    assert proof['passed'] and proof['remote'] == OLD_REMOTE and len(proof['targets']) == 33
    assert proof['collection'] == pin(OLD / 'capture-collected/capture-collection.json')
    assert proof['transfer'] == pin(OLD / 'capture-transfer.json')
    for name, wanted in proof['targets'].items():
        assert pin(OLD / 'capture-collected' / name) == wanted
    stage, remote = paths('stages')
    added = remote + '/spec.json'
    wanted = pin(stage / 'bundle/spec.json')
    assert wanted['sha256'] == '30e7845d6f8c9f2af5e8a4a9995ef508ae34a50208b827aafe8243fd54e77bec'
    assert added not in proof['manifests']
    recovery = dict(prior=pin(out / 'prospective.json'), added_manifest={added:wanted},
        reason='Staging added its specification between census and final recheck; final guard refused before deletion',
        helper=pin(__file__))
    write(out / 'recovery-prospective.json', recovery)
    owners = read(OLD / 'capture-collected/capture-collection.json')['identities']
    result = ssh(original.PRELUDE + f'''
from remote import idle,live,verify,pin,read
idle();verify();assert psutil.boot_time()==1789634288.0
owners={owners!r};assert not any(live(i) for i in owners)
manifests={proof['manifests']!r};manifests[{added!r}]={wanted!r}
assert set(manifests)=={{str(p) for d in Path('/dev/shm').glob('lokad-*') for n in ['spec.json','stage.json','payload.json'] if (p:=d/n).exists()}}
for name,wanted in manifests.items():assert pin(name)==wanted,name
targets={proof['targets']!r}
new=read(Path({added!r}));protected={{str((Path({remote!r})/n).resolve()) for n in new['files']}}
protected.update(str(Path(n).resolve()) for n in new['external'])
assert not any(str(base/n) in protected for n in targets)
for name,wanted in targets.items():
 p=base/name
 assert base.resolve()==base and base.parent==Path('/dev/shm')
 assert p.resolve()==p and p.is_relative_to(base) and p.parent==base/'stages'
 assert not p.is_symlink() and p.stat().st_nlink==1 and pin(p)==wanted
assert not (Path({remote!r})/'capture-state.json').exists()
physical=sum((base/n).stat().st_blocks*512 for n in targets)
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)
for name in targets:(base/name).unlink()
assert all(not (base/n).exists() for n in targets)
print(json.dumps(dict(passed=True,files=len(targets),physical_bytes=physical,before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free))))
''')
    for name,wanted in proof['targets'].items():
        assert pin(OLD / 'capture-collected' / name) == wanted
    write(out / 'closed.json', dict(**result, prospective=pin(out / 'prospective.json'),
        recovery=pin(out / 'recovery-prospective.json'), initial_refusal_retained=True, all_local_originals_retained=True))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
