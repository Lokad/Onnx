"""Find exact local originals for optional retirement of closed VM output copies."""
import ast
import json
from pathlib import Path
from retire_closed_output_duplicates import ROOT, CANDIDATES, ELIGIBLE, allowed, pin, read


def main():
    assert not CANDIDATES.exists()
    eligible = read(ELIGIBLE)
    assert eligible['passed'] and eligible['read_only'] and not eligible['removed']
    rows = []
    skipped = []
    count = 0
    for folder in (ROOT / 'artifacts').iterdir():
        if not folder.is_dir() or not all((folder / n).is_file() for n in ['closed.json', 'prepared.json', 'collected/collection.json']):
            continue
        receipt = read(folder / 'collected/collection.json')
        proof = read(folder / 'closed.json')
        if not (receipt.get('terminal') and receipt.get('code') == 0 and proof.get('passed')):
            continue
        if not isinstance(receipt.get('files'), dict) or not isinstance(proof.get('files'), dict):
            skipped.append([folder.name, 'unsupported collection or closure format'])
            continue
        names = [n for n, value in receipt['files'].items() if allowed(Path(n)) and value['bytes'] > 0]
        if not names:
            continue
        prepared = read(folder / 'prepared.json')
        if not isinstance(prepared.get('files'), dict):
            skipped.append([folder.name, 'unsupported preparation format'])
            continue
        remotes = set()
        sources = {}
        for name, wanted in prepared['files'].items():
            if not name.startswith('tests/') or not name.endswith('/run.py'):
                continue
            path = ROOT / name
            if not path.is_file() or pin(path) != wanted:
                continue
            tree = ast.parse(path.read_text(encoding='utf8'))
            constants = [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            if 'artifacts/' + folder.name not in constants:
                continue
            for statement in tree.body:
                if (isinstance(statement, ast.Assign)
                    and any(isinstance(n, ast.Name) and n.id == 'REMOTE' for n in statement.targets)
                    and isinstance(statement.value, ast.Constant) and isinstance(statement.value.value, str)
                    and statement.value.value.startswith('/dev/shm/lokad-')):
                    remotes.add(statement.value.value)
                    sources[name] = wanted
        if len(remotes) != 1:
            skipped.append([folder.name, 'no unique exact-source namespace'])
            continue
        remote, = remotes
        names = [n for n in names if remote + '/' + n in eligible['files']]
        if not names:
            continue
        receipt_local = folder / 'collected/collection.json'
        receipt_key = 'collected/collection.json'
        if receipt_key not in proof['files']:
            receipt_key = receipt_local.relative_to(ROOT).as_posix()
        if receipt_key not in proof['files']:
            skipped.append([folder.name, 'unrecognized receipt paths'])
            continue
        assert pin(receipt_local) == proof['files'][receipt_key]
        closure_pin = pin(folder / 'closed.json')
        receipt_pin = pin(receipt_local)
        for name in names:
            local = folder / 'collected' / name
            key = 'collected/' + name
            if key not in proof['files']:
                key = local.relative_to(ROOT).as_posix()
            if key not in proof['files']:
                continue
            if not local.is_file():
                skipped.append([folder.name + '/' + name, 'no retained local original'])
                continue
            wanted = pin(local)
            assert wanted == receipt['files'][name] == proof['files'][key]
            rows.append(dict(root=remote, name=name, local=local.relative_to(ROOT).as_posix(), identity=wanted,
                closure=closure_pin, collection=receipt_pin, owners=receipt['identities'], sources=sources))
        count += 1
        if count % 20 == 0:
            print(json.dumps(dict(verified_folders=count, candidate_files=len(rows))), flush=True)
    assert len({r['root'] + '/' + r['name'] for r in rows}) == len(rows)
    with CANDIDATES.open('x', encoding='utf8') as stream:
        json.dump(dict(local_only=True, remote_inspected=False, removed=False, candidates=rows, skipped=skipped,
                       generator=pin(Path(__file__)), eligible=pin(ELIGIBLE)), stream, indent=2)
    print(json.dumps(dict(candidates=len(rows), logical_bytes=sum(r['identity']['bytes'] for r in rows),
        roots=len({r['root'] for r in rows}), manifest_bytes=CANDIDATES.stat().st_size, skipped=len(skipped))))


if __name__ == '__main__':
    main()
