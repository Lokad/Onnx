"""Classify changed paths for CI. Unknown files conservatively run all jobs."""
import argparse
import json
from pathlib import Path
import subprocess


def classify(paths):
    paths = list(paths)
    full = any(not path.lower().endswith('.md') for path in paths)
    package = full or any(path in ('README.md', 'CHANGELOG.md') for path in paths)
    return {'tests': full, 'package': package}


def changed_paths(base, head):
    if not base or set(base) == {'0'}:
        return ['initial-push']
    # A shallow Actions checkout fetches these exact commits before invoking us.
    output = subprocess.check_output(['git', 'diff', '--name-only', '-z', '--no-renames', base, head])
    return output.decode('utf-8').rstrip('\0').split('\0') if output else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--head', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = classify(changed_paths(args.base, args.head))
    with args.output.open('a', encoding='utf-8') as stream:
        for key, value in result.items():
            stream.write(f'{key}={str(value).lower()}\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
