"""Retire two terminal builds' regenerable NuGet caches, preserving pinned feed/output."""
import json
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/parakeet/pad-dispatch-build-amd'))
import run
from protocol import pin, read, save

BASE = ROOT / 'artifacts/parakeet-padding-package-retention-20260923'
NAMES = ['lokad-parakeet-last-axis-pad-build-20260923', 'lokad-parakeet-pad-dispatch-build-20260923']
PROOFS = [
    ('parakeet-last-axis-pad-build-amd-20260923', '62043d100a96d418b02e25d09a65d8f89655ea21147c27d15a5efd861cce7e1c'),
    ('parakeet-pad-dispatch-build-amd-20260923', '7fbee9acdcedfa3e86ad32b573c36a5bbd493be050d798509b0afd1acc80c9db')]


def main():
    assert not BASE.exists()
    for name, digest in PROOFS:
        folder = ROOT / 'artifacts' / name
        assert pin(folder / 'closed.json')['sha256'] == digest
        proof = read(folder / 'closed.json'); assert proof['passed']
        for name, wanted in proof['files'].items(): assert pin(folder / name) == wanted, name
    BASE.mkdir()
    script = run.PRELUDE + textwrap.dedent(f'''
        from protocol import read,pin
        from remote import idle,live
        idle();assert psutil.boot_time()==1789634288.0
        names={NAMES!r}
        protected=set()
        for folder in Path('/dev/shm').glob('lokad-*'):
            if not (folder/'payload.json').exists():continue
            v=read(folder/'payload.json')
            protected.update(str((folder/n).resolve()) for n in v.get('files',{{}}))
            protected.update(str(Path(n).resolve()) for n in v.get('external',{{}}))
        owners=[];files={{}};archives={{}};roots=[]
        for name in names:
            root=Path('/dev/shm')/name
            receipt=read(root/'collection.json')
            assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
            assert not any(live(i) for i in receipt['identities']);owners.extend(receipt['identities'])
            payload=read(root/'payload.json')
            for n,w in payload['files'].items():assert pin(root/n)==w,n
            for n,w in payload['external'].items():assert pin(Path(n))==w,n
            feed=Path(payload['feed'])
            target=(root/'packages').resolve()
            assert target.parent==root.resolve() and target.name=='packages' and target.is_dir() and not target.is_symlink()
            roots.append(str(target))
            for p in target.rglob('*'):
                assert not p.is_symlink(),str(p)
                if not p.is_file():continue
                assert p.resolve().is_relative_to(target) and str(p.resolve()) not in protected
                files[str(p.resolve())]=pin(p)
            packages=list(target.rglob('*.nupkg'));assert packages
            for package in packages:
                retained=feed/package.name
                assert retained.is_file() and pin(package)==pin(retained),str(package)
                assert str(retained.resolve()) in protected
                archives[str(package)]=dict(retained=str(retained),identity=pin(retained))
        print(json.dumps(dict(passed=True,roots=roots,owners=owners,files=files,archives=archives,
            before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
    ''')
    snapshot = json.loads(run.ssh(script, 300)); save(BASE / 'snapshot.json', snapshot)
    assert snapshot['passed'] and snapshot['files'] and snapshot['archives']
    script = run.PRELUDE + textwrap.dedent(f'''
        import shutil
        from protocol import read,pin
        from remote import idle,live
        idle();assert psutil.boot_time()==1789634288.0
        value={snapshot!r}
        assert not any(live(i) for i in value['owners'])
        protected=set()
        for folder in Path('/dev/shm').glob('lokad-*'):
            if not (folder/'payload.json').exists():continue
            p=read(folder/'payload.json')
            protected.update(str((folder/n).resolve()) for n in p.get('files',{{}}))
            protected.update(str(Path(n).resolve()) for n in p.get('external',{{}}))
        for n,w in value['files'].items():assert n not in protected and pin(Path(n))==w,n
        for a in value['archives'].values():assert pin(Path(a['retained']))==a['identity']
        for target_name in value['roots']:
            target=Path(target_name).resolve()
            assert str(target) in [str(Path('/dev/shm')/n/'packages') for n in {NAMES!r}]
            actual={{str(p.resolve()) for p in target.rglob('*') if p.is_file()}}
            expected={{n for n in value['files'] if Path(n).is_relative_to(target)}}
            assert actual==expected and not target.is_symlink()
            shutil.rmtree(target)
        for name in {NAMES!r}:
            root=Path('/dev/shm')/name;p=read(root/'payload.json')
            for n,w in p['files'].items():assert pin(root/n)==w,n
            for n,w in p['external'].items():assert pin(Path(n))==w,n
        print(json.dumps(dict(passed=True,files=len(value['files']),bytes=sum(w['bytes'] for w in value['files'].values()),
            cached_archives=len(value['archives']),before=value['before'],
            after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
    ''')
    result = json.loads(run.ssh(script, 300))
    save(BASE / 'closed.json', dict(**result, snapshot=pin(BASE / 'snapshot.json'),
        generator=pin(Path(__file__)), scope='Generated extracted caches only; every cached nupkg matches a protected offline-feed archive. No source, measured binary, result, model or immutable payload path retired.'))
    print(json.dumps(result))


if __name__ == '__main__':
    main()
