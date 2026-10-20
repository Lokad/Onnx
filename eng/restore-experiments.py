"""Restore archived sources from local Git history into a new isolated directory."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess


def restore(repo, manifest, destination, prefix):
    repo, destination = repo.resolve(), destination.resolve()
    if destination.exists() or destination.is_relative_to(repo / '.git'):
        raise ValueError('choose a new destination outside .git; existing files are never overwritten')
    commit = manifest['commit']
    if len(commit) != 40 or any(c not in '0123456789abcdef' for c in commit):
        raise ValueError('archive commit must be a full Git commit ID')
    try:
        tree = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '-z', commit], stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as error:
        raise ValueError(f'archive commit missing locally; fetch its history first: git fetch origin {commit}') from error
    objects = {}
    for item in tree.split(b'\0'):
        if item:
            info, name = item.split(b'\t', 1)
            mode, kind, oid = info.decode().split()
            objects[name.decode('utf-8')] = (mode, kind, oid)
    selected = [name for name in manifest['paths'] if name.startswith(prefix)]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError('archive selection is empty or contains duplicate paths')
    for name in selected:
        if not name or any(c in name for c in ('\\', ':', '\0', '\n', '\r')) or PurePosixPath(name).is_absolute() or '..' in PurePosixPath(name).parts:
            raise ValueError('unsafe archive path: ' + name)
        if name not in objects or objects[name][:2] not in (('100644', 'blob'), ('100755', 'blob')):
            raise ValueError('archive path is missing or is not a regular file: ' + name)
    destination.mkdir(parents=True)
    with subprocess.Popen(['git', '-C', str(repo), 'cat-file', '--batch'], stdin=subprocess.PIPE, stdout=subprocess.PIPE) as process:
        for name in selected:
            process.stdin.write((objects[name][2] + '\n').encode())
            process.stdin.flush()
            oid, kind, size = process.stdout.readline().decode().split()
            if oid != objects[name][2] or kind != 'blob':
                raise ValueError('unexpected Git object: ' + name)
            content = process.stdout.read(int(size))
            if len(content) != int(size) or process.stdout.read(1) != b'\n':
                raise ValueError('incomplete Git object: ' + name)
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open('xb') as stream:
                stream.write(content)
            if objects[name][0] == '100755':
                target.chmod(0o755)
        process.stdin.close()
        if process.wait() != 0:
            raise ValueError('git cat-file failed; partial extraction remains for inspection')
    return len(selected)


def main():
    repo = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prefix', default='', help='restore only archived paths with this prefix')
    args = parser.parse_args()
    try:
        count = restore(repo, json.loads((repo / 'archives/experiments.json').read_text()), args.output, args.prefix)
        print(f'Restored {count} archived files to {args.output.resolve()}')
    except (OSError, ValueError) as error:
        parser.exit(1, str(error) + '\n')


if __name__ == '__main__':
    main()
