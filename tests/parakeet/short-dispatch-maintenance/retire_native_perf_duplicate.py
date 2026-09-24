"""Retire one closed VM perf copy; keep its verified local raw file and archive."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'tests/parakeet/validated-composition-pyannote-amd'))
from run import ssh, PRELUDE
from protocol import pin, read, save

SOURCE = ROOT/'artifacts/parakeet-ort-native-samples-20260924'
OUT = ROOT/'artifacts/parakeet-native-perf-remote-retention-20260924'
REMOTE = '/dev/shm/lokad-parakeet-ort-native-samples-20260924'


def main():
    assert not OUT.exists()
    proof = read(SOURCE/'closed.json')
    assert proof['passed'] and pin(SOURCE/'closed.json')['sha256'] == '546b3a58af8814772f0a5b816dbcbece710facc01980acebeb2fc93d7f0dcca9'
    transfer = read(SOURCE/'transfer.json')
    assert transfer['passed'] and pin(SOURCE/'transfer.json') == proof['transfer']
    assert pin(SOURCE/'results.tar.gz') == transfer['archive']
    assert pin(SOURCE/'terminal.json') == transfer['terminal']
    terminal = read(SOURCE/'terminal.json')
    assert terminal['state']['complete'] and terminal['state']['code'] == 0
    for name, wanted in terminal['files'].items():
        assert pin(SOURCE/'collected'/name) == wanted, name
    wanted = terminal['files']['perf.data']
    OUT.mkdir()
    script = PRELUDE + f'''
from protocol import pin,read
from remote import idle,live
idle(); assert psutil.boot_time()==1789634288.0
root=Path({REMOTE!r}); target=root/'perf.data'
assert root.resolve().parent==Path('/dev/shm') and target.resolve()==target
assert not target.is_symlink() and target.stat().st_nlink==1
assert pin(root/'state.json')=={terminal['files']['state.json']!r}
assert not any(live(i) for i in {proof['terminal_owners']!r})
protected=set()
for folder in Path('/dev/shm').glob('lokad-*'):
 for manifest in ['payload.json','stage.json']:
  if not (folder/manifest).exists():continue
  value=read(folder/manifest)
  protected.update(str((folder/n).resolve()) for n in value.get('files',{{}}))
  protected.update(str(Path(n).resolve()) for n in value.get('external',{{}}))
assert str(target) not in protected and pin(target)=={wanted!r}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(root).free)
'''
    snapshot = json.loads(ssh(script + 'print(json.dumps(dict(passed=True,before=before,protected_paths=len(protected))))'))
    save(OUT/'prepared.json', dict(**snapshot, target=REMOTE+'/perf.data', retained=(SOURCE/'collected/perf.data').relative_to(ROOT).as_posix(),
        identity=wanted, closure=pin(SOURCE/'closed.json'), archive=transfer['archive'], generator=pin(Path(__file__))))
    result = json.loads(ssh(script + '''
target.unlink(); assert not target.exists()
print(json.dumps(dict(passed=True,bytes_retired={wanted['bytes']},before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(root).free))))
'''.replace("{wanted['bytes']}", str(wanted['bytes']))))
    assert pin(SOURCE/'collected/perf.data') == wanted
    save(OUT/'closed.json', dict(**result, prepared=pin(OUT/'prepared.json'), local_raw_retained=True, local_archive_retained=True))
    print(json.dumps(result))


if __name__ == '__main__': main()
