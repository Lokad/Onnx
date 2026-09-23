"""Keep complete bodies and compare matched Tier1 instruction text."""
import difflib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-isolated-runtime-diagnostic-amd-20260923'
OUT = ROOT / 'artifacts/parakeet-isolated-runtime-codegen-20260923'
sys.path.insert(0, str(ROOT / 'tests/parakeet/isolated-runtime-diagnostic-amd'))
from protocol import pin, read
from checks import codegen_bodies


def parse(path):
    text = path.read_text()
    matches = list(re.finditer(r'^; Assembly listing for method ([^\n]+) \(([^\n]+)\)\n(.*?); Total bytes of code (\d+)', text, re.M | re.S))
    return [dict(index=i, method=m[1], tier=m[2], bytes=int(m[4]), body=m[0],
        instructions=[line.strip() for line in m[3].splitlines() if re.match(r'^       [a-z]', line)]) for i, m in enumerate(matches)]


def normalized(row):
    labels = dict(re.findall(r'^(G_M\d+_IG\d+):\s+;; offset=(0x[0-9A-Fa-f]+)', row['body'], re.M))
    result = []
    for instruction in row['instructions']:
        instruction = re.sub(r'0x[0-9A-Fa-f]{10,}', '<address>', instruction)
        instruction = re.sub(r'G_M\d+_IG\d+', lambda m: labels[m[0]], instruction)
        instruction = re.sub(r'for IG\d+', 'padding', instruction)
        result.append(instruction)
    return result


def main():
    assert not OUT.exists()
    closure = read(BASE / 'closed.json'); assert closure['passed'] and closure['diagnostic_only']
    for n, v in closure['files'].items(): assert pin(BASE / n) == v, n
    OUT.mkdir(); roles = {}; inputs = {}; bodies = []; comparisons = []
    for role in ['current', 'candidate']:
        path = BASE / 'collected/logs' / (role + '-codegen.stdout'); inputs[role] = pin(path)
        roles[role] = parse(path)
        census = codegen_bodies(path.read_text(), role)
        assert census == read(BASE / 'collected' / (role + '-codegen/bodies.json'))
        for row, record in zip(roles[role], census, strict=True):
            assert [row[k] for k in ['index', 'method', 'tier', 'bytes']] == [record[k] for k in ['index', 'method', 'tier', 'bytes']]
            target = OUT / (role + '-' + str(row['index']) + '.txt')
            target.write_text(row['body'] + '\n')
            bodies.append(dict(role=role, **record, file=target.name, identity=pin(target)))
    for left in roles['current']:
        if left['tier'] != 'Tier1': continue
        matches = [r for r in roles['candidate'] if r['method'] == left['method'] and r['tier'] == 'Tier1']
        if len(matches) != 1: continue
        right = matches[0]; before = normalized(left); after = normalized(right)
        target = OUT / ('tier1-' + str(left['index']) + '-' + str(right['index']) + '.diff')
        target.write_text('\n'.join(difflib.unified_diff(before, after, fromfile='current', tofile='candidate', lineterm='')) + '\n')
        comparisons.append(dict(method=left['method'], current=left['index'], candidate=right['index'],
            current_bytes=left['bytes'], candidate_bytes=right['bytes'], instruction_text_equal=before == after,
            diff=target.name, identity=pin(target)))
    result = dict(diagnostic_only=True, closure=pin(BASE / 'closed.json'), inputs=inputs, generator=pin(Path(__file__)),
        bodies=bodies, comparisons=comparisons,
        scope='Independent disassembly processes. Resolve internal branch offsets; mask hexadecimal values of at least ten digits and normalize alignment comments. Preserve instructions and padding counts. Text equality is not binary equality or proof of the version executed in a traced/scored call.')
    target = TOOLS / 'codegen-review-20260923.json'
    with target.open('x', encoding='utf8') as f: f.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(bodies=len(bodies), comparisons=comparisons, review=pin(target))))


if __name__ == '__main__': main()
