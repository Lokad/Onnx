"""Retire only the three terminal VM trace duplicates after local hash verification."""
from pathlib import Path
import json
from run import BASE, ROOT, PRELUDE, pin, read, write, ssh


def main():
    proof = read(BASE/'closed.json'); assert proof['passed']
    receipt = read(BASE/'collected/collection.json')
    assert pin(BASE/'collected/collection.json') == proof['collection']
    observation = read(BASE/'collected/profile/observation.json')
    files = {'profile/'+p['file']: {k:p[k] for k in ['bytes','sha256']} for p in observation['profiles'].values()}
    assert len(files) == 3
    for name, wanted in files.items():
        assert pin(BASE/'collected'/name) == receipt['files'][name] == wanted
    assert pin(BASE/'results.tar.gz') == read(BASE/'transfer.json')['archive']
    out = ROOT/'artifacts/parakeet-ort-remote-profile-retention-20260924'
    assert not out.exists(); out.mkdir()
    write(out/'prepared.json', dict(closure=pin(BASE/'closed.json'), files=files, source=pin(__file__)))
    result = ssh(PRELUDE+f'''
sys.path.insert(0,str(base))
from remote import live,pin,read
assert read(base/'state.json')['complete'] and all(not live(i) for i in {proof['terminal_owners']!r})
files={files!r}
for name,wanted in files.items():
 path=(base/name).resolve()
 assert path.parent==(base/'profile').resolve() and path.suffix=='.json' and pin(path)==wanted
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free)
for name in files:(base/name).unlink()
print(json.dumps(dict(passed=True,retired=files,before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage(base).free))))
''')
    write(out/'closed.json', result)
    print(json.dumps(dict(passed=True,bytes=sum(v['bytes'] for v in files.values()),**{k:result[k] for k in ['before','after']})))


if __name__ == '__main__':
    main()
