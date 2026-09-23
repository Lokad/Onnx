"""Read-only logical file-size inventory; never traverse directory aliases."""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import stat
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional new JSON receipt within this repository.')
    args = parser.parse_args()
    if args.output is not None:
        args.output = args.output.resolve()
        assert args.output.is_relative_to(ROOT) and not args.output.exists()
        assert args.output.parent.is_dir()

    # The prefix permits inspection of retained deep build paths on Windows.
    root = '\\\\?\\' + str(ROOT) if os.name == 'nt' else str(ROOT)
    pending = [root]
    totals = defaultdict(int)
    counts = defaultdict(int)
    aliases = []
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                info = entry.stat(follow_symlinks=False)
                relative = Path(os.path.relpath(entry.path, root))
                if stat.S_ISLNK(info.st_mode) or getattr(info, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    target = (ROOT / relative).resolve()
                    aliases.append(dict(path=relative.as_posix(), target=str(target),
                                        within_repository=target.is_relative_to(ROOT)))
                elif stat.S_ISDIR(info.st_mode):
                    pending.append(entry.path)
                elif stat.S_ISREG(info.st_mode):
                    totals[relative.parts[0]] += info.st_size
                    counts[relative.parts[0]] += 1
                else:
                    raise AssertionError(('Unclassified filesystem entry', str(relative)))

    # A top-level directory alias can be accounted for without traversing it
    # again. Unknown/outside aliases stay explicit rather than guessed.
    for alias in aliases:
        target = Path(alias['target'])
        relative = target.relative_to(ROOT) if alias['within_repository'] else None
        alias['duplicate_bytes'] = totals.get(relative.parts[0]) if relative is not None and len(relative.parts) == 1 else None
    total = sum(totals.values())
    conservative = total + sum(a['duplicate_bytes'] for a in aliases) if all(a['duplicate_bytes'] is not None for a in aliases) else None
    value = dict(checked=time.time(), repo_bytes=total, repo_files=sum(counts.values()),
                 artifact_bytes=totals.get('artifacts', 0),
                 within_50_decimal_gb=total < 50_000_000_000,
                 counting_known_aliases_again_bytes=conservative,
                 totals=dict(sorted(totals.items())), counts=dict(sorted(counts.items())),
                 aliases=aliases,
                 accounting='Sum of regular file lengths, with hardlinked paths counted separately and reparse points excluded; not filesystem allocation. Optional output receipt is written after this inventory.')
    if args.output is not None:
        with args.output.open('x', encoding='utf8') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    print(json.dumps({k: v for k, v in value.items() if k not in ['totals', 'counts']}, indent=2))


if __name__ == '__main__':
    main()
