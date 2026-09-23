"""Compare common Tier1 text using resolved label offsets and masked addresses."""
import json
import re
from inspect_codegen import BASE, OUT, parse, pin


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
    output = OUT / 'resolved-shared-tier1-comparison.json'; assert not output.exists()
    roles = {}; inputs = {}
    for role in ['current', 'candidate']:
        path = BASE / 'collected/logs' / (role + '-codegen-512.stdout')
        roles[role] = parse(path); inputs[role] = pin(path)
    comparisons = []
    for old in roles['current']:
        if old['tier'] != 'Tier1': continue
        matches = [r for r in roles['candidate'] if r['method'] == old['method'] and r['tier'] == 'Tier1']
        if len(matches) != 1: continue
        new = matches[0]
        comparisons.append(dict(method=old['method'], current=old['index'], candidate=new['index'],
            current_bytes=old['bytes'], candidate_bytes=new['bytes'],
            instruction_text_equal=normalized(old) == normalized(new)))
    result = dict(inputs=inputs, comparisons=comparisons,
        scope='Resolved local branch offsets, preserved alignment-byte counts and masked hexadecimal values of at least ten digits. Instruction-text comparison only; not binary identity or a performance claim.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
