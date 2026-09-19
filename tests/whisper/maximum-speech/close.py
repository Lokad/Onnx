"""Close one successfully audited maximum-speech evidence directory exactly once."""
from pathlib import Path
import argparse
import json
import subprocess
import time
import psutil
from prepare import sha, read, write_new, pin


def absent(items):
    for item in items:
        try:
            assert psutil.Process(item['pid']).create_time() != item['create_time'], 'Owned local process remains'
        except psutil.NoSuchProcess:
            pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--windows',type=Path,required=True)
    parser.add_argument('--amd',type=Path,required=True)
    parser.add_argument('--kind',choices=('windows','amd'),required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    windows,amd = args.windows.resolve(),args.amd.resolve()
    base = windows if args.kind == 'windows' else amd
    assert not (base/'closed.json').exists()
    waudit,aaudit = read(windows/'audit.json'),read(amd/'audit.json')
    assert waudit['passed'] and aaudit['passed'] and aaudit['windows_audit_sha256'] == sha(windows/'audit.json')
    frozen,continuation = read(windows/'frozen.json'),read(windows/'continuation.json')
    for name,expected in frozen['files'].items():
        assert pin(windows/name) == expected,name
    for name,expected in continuation['files'].items():
        assert pin(windows/'continuation-source'/name) == expected,name
    assert waudit['auditor_sha256'] == sha(windows/'continuation-source/audit.py')
    assert aaudit['auditor_sha256'] == sha(amd/'payload/audit_amd.py')
    assert read(windows/'managed-process.json')['supervisor_sha256'] == sha(windows/'continuation-source/supervise.py')
    assert continuation['native_recovery_sha256'] == sha(windows/'native-terminal-recovery.json')
    assert waudit['native_sha256'] == sha(windows/'native/manifest.json')
    assert waudit['managed_sha256'] == sha(windows/'managed/result.json')
    assert aaudit['managed_sha256'] == sha(amd/'collected/result/managed/result.json')
    for name,item in read(windows/'native/manifest.json')['files'].items():
        assert pin(windows/'native'/name) == {key:item[key] for key in ('bytes','sha256')}
    for name,expected in read(amd/'payload/bundle.json')['files'].items():
        assert pin(amd/'payload'/name) == expected,name
    collection = read(amd/'collected/collection.json')
    for name,expected in collection['files'].items():
        assert pin(amd/'collected'/name) == expected,name
    absent([p for row in waudit['resources'] for p in row['terminal_processes']])
    remote_check = '''from pathlib import Path
import json
items=ITEMS
for item in items:
    p=Path('/proc')/str(item['pid'])/'stat'
    if p.exists():
        fields=p.read_text().split(') ',1)[1].split()
        assert int(fields[19])!=item['start'],item
print(json.dumps(dict(all_absent=True,identities=items)))
'''.replace('ITEMS',repr(aaudit['resources']['terminal_processes']))
    observed = subprocess.run(['ssh','-i','C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem','-o','BatchMode=yes',
        'vermorel@74.178.91.76','python3 -B -'],input=remote_check,text=True,capture_output=True,check=True)
    terminal = json.loads(observed.stdout)
    assert terminal['all_absent']
    files = {p.relative_to(base).as_posix():pin(p) for p in sorted(base.rglob('*')) if p.is_file()}
    tracked = {p.relative_to(root).as_posix():pin(p) for p in sorted(Path(__file__).parent.iterdir()) if p.suffix in ('.py','.md')}
    record = dict(schema=1,closed=True,closed_at=time.time(),kind=args.kind,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        files=files,tracked=tracked,windows_audit_sha256=sha(windows/'audit.json'),amd_audit_sha256=sha(amd/'audit.json'),
        native_exit_failure_preserved=True,native_terminal_recovery_sha256=sha(windows/'native-terminal-recovery.json'),
        all_owned_processes_terminal=True,remote_terminal_check=terminal,scope='Finite cyclic600-second API qualification on two hosts')
    if args.kind == 'amd':
        assert read(windows/'closed.json')['closed']
        record['windows_closed_sha256'] = sha(windows/'closed.json')
    write_new(base/'closed.json',record)
    print(json.dumps(dict(kind=args.kind,files=len(files),tracked=len(tracked),closed_sha256=sha(base/'closed.json')),indent=2))


if __name__ == '__main__':
    main()
