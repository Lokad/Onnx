"""Compress three closed native traces in place; preserve paths and decoded bytes."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'artifacts/parakeet-transpose-axis-ort-profile-recovery-amd-20260928'
OUT = ROOT/'artifacts/repository-retention-20260923/transpose-native-compression-20260928'
spec = importlib.util.spec_from_file_location('allocation', ROOT/'eng/dedupe-completed-parakeet-evidence.py')
allocation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(allocation)
pin, save, checked = allocation.pin, allocation.save, allocation.checked


def main():
    assert os.name == 'nt' and not OUT.exists()
    read = lambda path: json.loads(path.read_text(encoding='utf8'))
    closure = pin(BASE/'closed.json')
    assert closure['sha256'] == '59cc8dee51a7f5822bd64c73fbc3c81a8e100a63f82084130076df329d393734'
    proof = read(BASE/'closed.json')
    assert proof['passed'] and proof['analysis'] == pin(BASE/'analysis.json')
    assert proof['transfer'] == pin(BASE/'transfer.json')
    assert proof['collection'] == pin(BASE/'collected/collection.json')
    receipt = read(BASE/'collected/collection.json')
    assert receipt['terminal'] and receipt['code'] == 0
    allocated = allocation.allocator()
    rows = []
    for name, wanted in receipt['files'].items():
        if not (name.startswith('profile/') and '_2026-' in name and name.endswith('.json')):
            continue
        path = BASE/'collected'/name
        info = checked(path)
        assert path.resolve().is_relative_to(BASE.resolve()) and info.st_nlink == 1
        assert pin(path) == wanted
        rows.append(dict(name=name, identity=wanted, allocated=allocated(path),
                         inode=info.st_ino, device=info.st_dev))
    assert len(rows) == 3
    OUT.mkdir()
    save(OUT/'prepared.json', dict(closure=closure, files=rows, helper=pin(Path(__file__)),
         method='NTFS compression of terminal trace files; no logical byte or path changes'))
    after = 0
    with (OUT/'journal.jsonl').open('x', encoding='utf8') as journal:
        for row in rows:
            path = BASE/'collected'/row['name']
            assert pin(path) == row['identity'] and checked(path).st_ino == row['inode']
            result = subprocess.run(['compact.exe', '/C', '/Q', str(path)],
                capture_output=True, check=True, creationflags=subprocess.CREATE_NO_WINDOW)
            info = checked(path)
            assert (info.st_dev, info.st_ino) == (row['device'], row['inode'])
            assert pin(path) == row['identity']
            size = allocated(path); after += size
            journal.write(json.dumps(dict(name=row['name'], allocated=size, bytes_preserved=True))+'\n')
            journal.flush()
    assert pin(BASE/'closed.json') == closure
    before = sum(row['allocated'] for row in rows)
    # New artifacts may already inherit NTFS compression from their parent.
    # An unchanged allocation is a valid no-op, not additional reclaimed space.
    assert before >= after
    value = dict(passed=True, files=len(rows), allocated_before=before, allocated_after=after,
        reclaimed_allocated_bytes=before-after, all_paths_and_bytes_preserved=True,
        closure_unchanged=True, preparation=pin(OUT/'prepared.json'), journal=pin(OUT/'journal.jsonl'))
    save(OUT/'closed.json', value)
    print(json.dumps(value))


if __name__ == '__main__': main()
