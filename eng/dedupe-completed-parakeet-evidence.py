"""Consolidate identical large files in two fixed, completed local evidence families."""
from collections import defaultdict
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT/'artifacts'
OUT = ARTIFACTS/'repository-retention-20260923/pointwise-local-dedup-20260927'
PATTERNS = ['parakeet-pointwise*-20260927', 'parakeet-decoder-lstm-layout*-20260927']
SUFFIXES = {'.json', '.dll', '.pdb', '.npy', '.f32', '.bin'}


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def save(path, value):
    with path.open('x', encoding='utf8') as stream:
        json.dump(value, stream, indent=2); stream.write('\n')


def checked(path):
    absolute = path.absolute()
    assert absolute.resolve() == absolute and absolute.is_relative_to(ARTIFACTS)
    info = absolute.stat(follow_symlinks=False)
    assert stat.S_ISREG(info.st_mode) and not info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT
    assert info.st_ino and info.st_dev
    return info


class StandardInfo(ctypes.Structure):
    _fields_ = [('allocated', ctypes.c_longlong), ('logical', ctypes.c_longlong),
                ('links', wintypes.DWORD), ('delete_pending', ctypes.c_ubyte), ('directory', ctypes.c_ubyte)]


def allocator():
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    create = kernel.CreateFileW
    create.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p,
                       wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    query = kernel.GetFileInformationByHandleEx
    query.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    query.restype = wintypes.BOOL
    close = kernel.CloseHandle; close.argtypes = [wintypes.HANDLE]; close.restype = wintypes.BOOL

    def allocated(path):
        handle = create('\\\\?\\'+str(path), 0x80, 7, None, 3, 0, None)
        assert handle != ctypes.c_void_p(-1).value, (path, ctypes.get_last_error())
        value = StandardInfo()
        try:
            assert query(handle, 1, ctypes.byref(value), ctypes.sizeof(value)), ctypes.get_last_error()
        finally:
            assert close(handle)
        assert not value.directory and not value.delete_pending and value.allocated >= 0
        assert value.logical == path.stat().st_size
        return value.allocated
    return allocated


def main():
    assert os.name == 'nt' and not OUT.exists()
    groups = defaultdict(list); closures = {}; declared = {}; allocated = allocator()
    for pattern in PATTERNS:
        for folder in sorted(ARTIFACTS.glob(pattern)):
            closure = folder/'closed.json'
            if not closure.is_file() or folder.is_symlink():
                continue
            checked(closure)
            value = json.loads(closure.read_text(encoding='utf8'))
            if not (value.get('passed') or value.get('completed')):
                continue
            closures[str(closure.relative_to(ROOT))] = pin(closure)
            for relative, wanted in value.get('files', {}).items():
                if not isinstance(wanted, dict) or wanted.get('bytes', 0) < 1_048_576:
                    continue
                path = folder/relative
                if path.suffix not in SUFFIXES:
                    continue
                info = checked(path)
                assert path.resolve().is_relative_to(folder.resolve()) and info.st_size == wanted['bytes']
                if info.st_nlink != 1:
                    continue
                key = (wanted['bytes'], wanted['sha256'], info.st_mode, info.st_file_attributes, info.st_dev)
                declared[path] = wanted
                groups[key].append(path)
    selected = [paths for paths in groups.values() if len(paths) > 1]
    paths = [path for group in selected for path in group]
    assert len(paths) == len(set(paths))
    identities = {}; records = []
    for group in selected:
        source = group[0]; wanted = declared[source]
        for path in group:
            assert pin(path) == wanted, path
            info = checked(path)
            identities[str(path.relative_to(ROOT))] = dict(pin=wanted, device=info.st_dev,
                inode=info.st_ino, attributes=info.st_file_attributes, mode=info.st_mode,
                allocated=allocated(path))
        records += [dict(source=str(source.relative_to(ROOT)), target=str(path.relative_to(ROOT)), pin=wanted)
                    for path in group[1:]]
    OUT.mkdir()
    save(OUT/'prospective.json', dict(closures=closures, files=identities, links=records,
        local_only=True, active_campaigns_excluded=True, minimum_bytes=1_048_576,
        policy='Closed evidence remains immutable; rebuild or modify a separate copy. Every path and content byte is retained.'))
    before = sum(row['allocated'] for row in identities.values())
    with (OUT/'journal.jsonl').open('x', encoding='utf8') as journal:
        for row in records:
            source = ROOT/row['source']; target = ROOT/row['target']
            a, b = checked(source), checked(target)
            assert a.st_dev == b.st_dev and a.st_mode == b.st_mode and a.st_file_attributes == b.st_file_attributes
            assert b.st_nlink == 1 and b.st_ino == identities[row['target']]['inode']
            assert pin(source) == pin(target) == row['pin']
            temporary = target.with_name(target.name+'.verified-local-link')
            assert not temporary.exists() and temporary.resolve().is_relative_to(ARTIFACTS)
            journal.write(json.dumps(dict(action='begin', **row))+'\n'); journal.flush()
            os.link(source, temporary)
            assert pin(temporary) == row['pin']
            checked(target); checked(temporary)
            os.replace(temporary, target)
            assert os.path.samefile(source, target) and pin(target) == row['pin']
            journal.write(json.dumps(dict(action='complete', **row))+'\n'); journal.flush()
    unique = {}
    for name, row in identities.items():
        path = ROOT/name; info = checked(path)
        assert pin(path) == row['pin']
        assert info.st_mode == row['mode'] and info.st_file_attributes == row['attributes']
        unique.setdefault((info.st_dev, info.st_ino), allocated(path))
    for name, wanted in closures.items():
        assert pin(ROOT/name) == wanted
    after = sum(unique.values())
    result = dict(passed=True, local_only=True, closed_directories=len(closures),
        duplicate_groups=len(selected), verified_files=len(identities), links=len(records),
        allocated_before=before, allocated_after=after, reclaimed_allocated_bytes=before-after,
        all_paths_and_bytes_preserved=True, closures_unchanged=True, completed=time.time(),
        prospective=pin(OUT/'prospective.json'), journal=pin(OUT/'journal.jsonl'), script=pin(Path(__file__)))
    assert before > after
    save(OUT/'closed.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
