"""Bind the case/count correction to the existing uninstrumented benchmark."""
from pathlib import Path
import hashlib

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'e5-warmed-qualification-amd'
NAMES = ['Program.cs', 'native.py', 'statistics.py', 'test_statistics.py',
         'checks.py', 'protocol.py', 'remote.py', 'run.py', 'audit.py', 'test_inventory.py']


def expected(name):
    source = ROOT/'tests/benchmarks/warmed-release-amd-v2/Program.cs' if name == 'Program.cs' else PARENT/name
    text = source.read_text(encoding='utf8')
    if name == 'Program.cs':
        assert text.count('3 : 780') == text.count('index < 600') == 1
        text = text.replace('3 : 780', '3 : 6180').replace('index < 600', 'index < 6000')
    elif name in ['native.py', 'statistics.py', 'test_statistics.py', 'checks.py']:
        assert '1380' in text and '1200' in text
        text = text.replace('1380', '6180').replace('1200', '6000')
    elif name == 'protocol.py':
        assert text.count("CASES=['e5-30tok']") == 1
        text = text.replace("CASES=['e5-30tok']", "CASES=['e5-8tok']")
    elif name == 'audit.py':
        text = text.replace('e5-30tok', 'e5-8tok').replace('8289', '37089')
    elif name == 'run.py':
        text = text.replace('e5-warmed-qualification-amd-20260924', 'e5-steady-short-amd-20260925')
        text = text.replace('lokad-e5-warmed-qualification-20260924', 'lokad-e5-steady-short-20260925')
        text = text.replace('def observe():\n', "def observe():\n    assert not (BASE/'closed.json').exists(), 'Preserve the closed campaign'\n")
    elif name == 'test_inventory.py':
        text = text.replace('64050000', '24180000').replace('B0040000', '70170000')
        text = text.replace('1380', '6180').replace('1200', '6000')
        text = text.replace('len(verify()),7', 'len(verify()),10')
    else:
        assert name == 'remote.py'
    return source, text


def verify():
    files = {}
    for name in NAMES:
        source, wanted = expected(name)
        assert (TOOLS/name).read_text(encoding='utf8') == wanted, name
        with source.open('rb') as stream:
            files[source.relative_to(ROOT).as_posix()] = dict(bytes=source.stat().st_size,
                sha256=hashlib.file_digest(stream, 'sha256').hexdigest())
    return files


if __name__ == '__main__':
    print(verify())
