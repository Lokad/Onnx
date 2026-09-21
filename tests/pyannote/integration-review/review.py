"""Inventory the candidate's real source changes without applying or running it."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-integration-review-20260921'
SOURCE = ROOT / 'artifacts/pyannote-sparse-mel-20260921/source'
PROOF = ROOT / 'artifacts/pyannote-sparse-mel-20260921/focused-closed.json'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf8')


def rel(path):
    return path.relative_to(ROOT).as_posix()


def main():
    assert not BASE.exists()
    assert pin(PROOF)['sha256'] == '35cba7a867ee2faff0fb8c3b94431c6f3ffa390648242188da2cd6515856cb6b'
    proof = json.loads(PROOF.read_text())
    assert proof['passed']
    files = {rel(PROOF): pin(PROOF), rel(Path(__file__)): pin(Path(__file__))}
    tracked = subprocess.check_output(['git', 'ls-files', '--', 'src'], cwd=ROOT, text=True).splitlines()
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    candidates = {p.relative_to(SOURCE).as_posix(): p for p in (SOURCE / 'src').rglob('*')
        if p.is_file() and p.suffix in ['.cs', '.csproj'] and not {'bin', 'obj'}.intersection(p.relative_to(SOURCE).parts)}
    assert all(name in candidates for name in tracked if Path(name).suffix in ['.cs', '.csproj'])
    rows, common, frontend = [], [], []
    for name, candidate in sorted(candidates.items()):
        assert proof['files'][rel(candidate)] == pin(candidate), name
        files[rel(candidate)] = pin(candidate)
        current = ROOT / name
        if current.exists():
            files[name] = pin(current)
        before = current.read_text(encoding='utf-8-sig') if current.exists() else ''
        after = candidate.read_text(encoding='utf-8-sig')
        byte_equal = current.exists() and pin(current) == pin(candidate)
        text_equal = before == after
        category = 'identical' if byte_equal else 'encoding-only' if text_equal else 'build-harness' if candidate.suffix == '.csproj' else 'sparse-mel' if candidate.name == 'WeSpeakerAudio.cs' else 'core-and-request-contexts'
        rows.append(dict(path=name, category=category, current_exists=current.exists(), current=pin(current) if current.exists() else None, candidate=pin(candidate)))
        if category in ['core-and-request-contexts', 'sparse-mel']:
            patch = ''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                fromfile='a/' + name if current.exists() else '/dev/null', tofile='b/' + name))
            (frontend if category == 'sparse-mel' else common).append(patch)
    BASE.mkdir()
    for name, patches in [('core-and-request-contexts.patch', common), ('sparse-mel.patch', frontend)]:
        path = BASE / name
        path.write_text(''.join(patches), encoding='utf8')
        check = subprocess.run(['git', 'apply', '--check', '--ignore-space-change', str(path)], cwd=ROOT,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        save(BASE / (name + '.check.json'), dict(code=check.returncode, output=check.stdout,
            scope='Checks applicability only; no patch applied, build or inference.'))
        assert check.returncode == 0, check.stdout
    counts = {category: sum(row['category'] == category for row in rows) for category in sorted({row['category'] for row in rows})}
    result = dict(passed=True, root_commit=head, source=rel(SOURCE), counts=counts, rows=rows,
        scope='Source integration inventory only. Common candidate still awaits target AMD qualification. Sparse mel additionally awaits its own full application/timing result. No production change, new build, inference or performance admission.')
    save(BASE / 'analysis.json', result)
    for path in BASE.iterdir():
        if path.is_file():
            files[rel(path)] = pin(path)
    for name, wanted in files.items():
        assert pin(ROOT / name) == wanted, name
    save(BASE / 'closed.json', dict(passed=True, files=files, analysis=pin(BASE / 'analysis.json'), scope=result['scope']))
    print(json.dumps(dict(counts=counts, closure=pin(BASE / 'closed.json'))))


if __name__ == '__main__':
    main()
