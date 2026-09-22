"""Correct display-only method labels; preserve the original report and all data."""
from pathlib import Path
import json
import hashlib

HERE = Path(__file__).resolve().parent


def pin(path):
    return dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    original = HERE/'results-20260922.md'
    raw = HERE/'observations-20260922.json'
    output = HERE/'results-v2-20260922.md'
    assert not output.exists()
    assert pin(original)['sha256'] == 'a0b074171f4dc6762887551cb5837138c6f3f4d0061740741456930fe5be9887'
    assert pin(raw)['sha256'] == '653ff30ec1619291aca509660e26c37ba04c0c99e2b518e5d23e59d17f48ddc1'
    data = json.loads(raw.read_text()); text = original.read_text(); changed = []
    for row in data['leaves'][:12]:
        before = row['method'].split('!')[-1].split('(')[0].replace('|', '\\|')
        after = row['method'].split('!', 1)[-1].split('(')[0].replace('|', '\\|')
        fence = '``' if '`' in after else '`'
        old = f'| `{before}` |'; new = f'| {fence}{after}{fence} |'
        assert text.count(old) == 1
        text = text.replace(old, new)
        if old != new: changed.append(dict(before=old, after=new))
    text += '\nDisplay correction: split the assembly separator only once so generic parameter markers do not truncate method names; use Markdown fences that preserve generic arity. The original report and every observation remain unchanged.\n'
    output.write_text(text, encoding='utf8')
    (HERE/'display-correction-20260922.json').write_text(json.dumps(dict(original=pin(original),
        observations=pin(raw), corrected=pin(output), changed=changed, data_unchanged=True), indent=2)+'\n')
    print(json.dumps(dict(corrected=pin(output), changed=changed)))


if __name__ == '__main__': main()
