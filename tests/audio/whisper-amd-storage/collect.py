"""Stream terminal evidence directly to local disk, retaining failed campaigns too."""
import tarfile
from common import *
from storage_contract import safe_member


def main():
    archive_path = BASE/'results.tar.gz'; target = BASE/'collected'
    assert not archive_path.exists() and not target.exists()
    frozen = pin(BASE/'frozen.json')
    script = PRELUDE+'''
assert pin(base/'frozen.json')==%r
state=read(base/'campaign/identity.json');assert state['complete'] is True
births=[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
terminal(births)
if (base/'collection.json').exists():
 receipt=read(base/'collection.json');assert receipt['frozen']==pin(base/'frozen.json')
 for name,wanted in receipt['files'].items():assert pin(base/name)==wanted,name
else:
 frozen=read(base/'frozen.json')
 for name,wanted in frozen['files'].items():assert pin(base/name)==wanted,name
 for name,wanted in frozen['external'].items():assert pin(name)==wanted,name
 paths=[p for p in sorted(base.rglob('*')) if p.is_file()]
 assert not any(p.is_symlink() for p in paths)
 files={p.relative_to(base).as_posix():pin(p) for p in paths}
 receipt=dict(terminal=True,code=state['code'],files=files,births=births,frozen=pin(base/'frozen.json'),external_verified=len(frozen['external']))
 write(base/'collection.json',receipt)
print(json.dumps(dict(receipt=pin(base/'collection.json'),files=len(receipt['files']),code=receipt['code'],births=births)))
''' % frozen
    snapshot = json.loads(ssh(script,timeout=1800)); write(BASE/'collection-snapshot.json',snapshot)
    script = PRELUDE+'''
import tarfile
assert pin(base/'collection.json')==%r
receipt=read(base/'collection.json');terminal(receipt['births'])
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name in [*receipt['files'],'collection.json']:archive.add(base/name,arcname=name,recursive=False)
''' % snapshot['receipt']
    with archive_path.open('xb') as output, (BASE/'transfer.stderr').open('x') as error:
        transfer = subprocess.run(SSH+['python3 -B -'],input=script.encode('utf8'),stdout=output,stderr=error,
            timeout=1800,creationflags=subprocess.CREATE_NO_WINDOW)
    assert transfer.returncode == 0
    target.mkdir()
    with tarfile.open(archive_path) as archive:
        members = archive.getmembers()
        assert len({m.name for m in members}) == len(members)
        for member in members:
            assert member.isfile(); safe_member(member.name)
        archive.extractall(target,filter='data')
    assert pin(target/'collection.json') == snapshot['receipt']
    receipt = read(target/'collection.json')
    for name, expected in receipt['files'].items():
        assert pin(target/name) == expected, name
    assert {p.relative_to(target).as_posix() for p in target.rglob('*') if p.is_file()} == set(receipt['files'])|{'collection.json'}
    assert pin(target/'frozen.json') == frozen
    ssh(PRELUDE+'terminal(%r)\n' % receipt['births'])
    write(BASE/'collection-transfer.json',dict(passed=True,archive=pin(archive_path),receipt=pin(target/'collection.json'),files=len(receipt['files']),code=receipt['code']))
    print(json.dumps(read(BASE/'collection-transfer.json')))


if __name__ == '__main__':
    main()
