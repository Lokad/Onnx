"""Retain complete local phase observations and retire only their idle VM copies."""
import json
from pathlib import Path
import retire_closed_output_duplicates as common

ROOT = common.ROOT
OUT = ROOT / 'artifacts/e5-repeatability-phase-retention-20260925'
TARGETS = [('parakeet-managed-phase-amd-20260924', 'lokad-parakeet-managed-phase-20260924'),
           ('parakeet-masking-padding-layout-amd-20260924', 'lokad-parakeet-masking-padding-layout-20260924')]
pin, read, save, ssh = common.pin, common.read, common.save, common.ssh


def main():
    assert not OUT.exists() and read(common.APP / 'closed.json')['admitted']
    eligible = read(common.ELIGIBLE)['files']
    roots = {}
    for local, remote in TARGETS:
        folder = ROOT / 'artifacts' / local
        proof = read(folder / 'closed.json')
        assert proof['passed'] and proof['analysis'] == pin(folder / 'analysis.json')
        collected = folder / 'capture-collected'
        assert proof['collection'] == pin(collected / 'capture-collection.json')
        receipt = read(collected / 'capture-collection.json')
        assert receipt['terminal'] and receipt['code'] == 0
        assert receipt['state'] == pin(collected / 'capture-state.json')
        state = read(collected / 'capture-state.json')
        assert state['complete'] and state['code'] == 0
        owners = [state['supervisor']] + [dict(pid=int(p), birth=b) for r in state['runs'] for p, b in r['members'].items()]
        assert owners == proof['terminal_owners']
        root = '/dev/shm/' + remote
        files = {}
        for name, wanted in receipt['files'].items():
            if root + '/' + name not in eligible or not common.allowed(Path(name)):
                continue
            assert pin(collected / name) == wanted
            files[name] = dict(identity=wanted, local=(collected / name).relative_to(ROOT).as_posix())
        assert files
        roots[root] = dict(files=files, collection=proof['collection'], owners=owners,
                           closure=pin(folder / 'closed.json'))
    script = common.COMMON + f'''
roots={roots!r}
for name,row in roots.items():
 root=Path(name);assert root.resolve()==root and root.parent==Path('/dev/shm')
 assert pin(root/'capture-collection.json')==row['collection'] and not any(live(i) for i in row['owners'])
 for relative,wanted in row['files'].items():
  path=root/relative
  assert path.resolve()==path and path.is_relative_to(root) and not path.is_symlink() and path.stat().st_nlink==1
  assert str(path) not in protected and allowed(path) and pin(path)==wanted['identity']
'''
    snapshot = json.loads(ssh(script + '''
print(json.dumps(dict(passed=True,manifests=manifests,
 physical_bytes=sum((Path(root)/name).stat().st_blocks*512 for root,r in roots.items() for name in r['files']),
 before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''', 300))
    OUT.mkdir()
    save(OUT / 'prospective.json', dict(**snapshot, roots=roots, generator=pin(Path(__file__)), common=pin(Path(common.__file__))))
    result = json.loads(ssh(script + f'''
assert manifests=={snapshot['manifests']!r}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
physical=sum((Path(root)/name).stat().st_blocks*512 for root,r in roots.items() for name in r['files'])
assert physical=={snapshot['physical_bytes']!r}
for root,row in roots.items():
 for name in row['files']:(Path(root)/name).unlink()
assert all(not (Path(root)/name).exists() for root,r in roots.items() for name in r['files'])
idle()
print(json.dumps(dict(passed=True,files=sum(len(r['files']) for r in roots.values()),physical_bytes=physical,
 logical_bytes=sum(w['identity']['bytes'] for r in roots.values() for w in r['files'].values()),before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''', 300))
    for row in roots.values():
        for wanted in row['files'].values():
            assert pin(ROOT / wanted['local']) == wanted['identity']
    save(OUT / 'closed.json', dict(**result, prospective=pin(OUT / 'prospective.json'), all_local_originals_retained=True))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
