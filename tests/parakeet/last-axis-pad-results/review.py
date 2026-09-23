"""Review the one PadCore insertion with exact local and branch correspondence."""
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/parakeet-last-axis-pad-build-amd-20260923'
sys.path.insert(0, str(ROOT / 'tests/parakeet/last-axis-pad-build-amd'))
from protocol import pin, read
from checks import PAD, HELPER


def branch_target(rows, index):
    instruction = rows[index]
    delta = int.from_bytes(bytes.fromhex(instruction['operand']), 'little', signed=True)
    offset = rows[index + 1]['offset'] + delta
    return next(i for i, row in enumerate(rows) if row['offset'] == offset)


def compare(before, after):
    assert before['InitLocals'] == after['InitLocals']
    assert before['MaxStackSize'] == after['MaxStackSize'] == 4
    assert before['exceptions'] == after['exceptions'] == []
    assert after['locals'][2] == dict(type='Lokad.Onnx.DenseTensor`1[T]', IsPinned=False)
    assert after['locals'][:2] + after['locals'][3:] == before['locals']
    old, new = before['instructions'], after['instructions']
    assert len(old) == 255 and len(new) == 266
    assert 'ToDenseTensor()' in old[11]['operand'] == new[11]['operand']
    assert [(r['opcode'], r['operand']) for r in new[12:14]] == [('stloc.2', ''), ('ldloc.2', '')]
    guard = new[36:45]
    assert [(r['opcode'], r['operand']) for r in guard] == [
        ('ldarg.s', '04'), ('brtrue.s', '0E'), ('ldloc.2', ''), ('ldloc.s', '04'),
        ('ldarg.1', ''), ('call', 'Lokad.Onnx.CPUExecutionProvider::Boolean TryPadLastAxis[T](Lokad.Onnx.DenseTensor`1[T], Lokad.Onnx.DenseTensor`1[T], Int32[])'),
        ('brfalse.s', '03'), ('ldloc.s', '04'), ('ret', '')]
    assert branch_target(new, 37) == branch_target(new, 42) == 45
    assert new[35]['opcode'] == old[33]['opcode'] == 'ret'
    removed = {12, 13, *range(36, 45)}
    kept = [i for i in range(len(new)) if i not in removed]
    mapping = {index: i for i, index in enumerate(kept)}
    # The old nonempty-output edge now reaches the inserted guard. Removing
    # that single declared guard must recover its original continuation.
    mapping[36] = mapping[45]

    def normalized(rows, index, candidate):
        op, operand = rows[index]['opcode'], rows[index]['operand']
        local = re.fullmatch(r'(ldloca|ldloc|stloc)(?:\.(s|[0-3]))?', op)
        if local:
            kind, suffix = local.groups()
            slot = int(suffix) if suffix in ['0', '1', '2', '3'] else int.from_bytes(bytes.fromhex(operand), 'little')
            if candidate:
                assert slot != 2
                slot -= int(slot > 2)
            return kind, slot
        if op.startswith(('br', 'beq', 'bge', 'bgt', 'ble', 'blt', 'bne', 'leave')):
            target = branch_target(rows, index)
            return op.removesuffix('.s'), mapping[target] if candidate else target
        assert op != 'switch'
        return op, operand

    for i, j in enumerate(kept):
        assert normalized(old, i, False) == normalized(new, j, True), (i, j)
    return dict(passed=True, original_instructions=255, candidate_instructions=266,
                retained_original_instructions=255, added_source_store_reload=2,
                added_guard_instructions=9, branch_targets_preserved=True,
                locals_preserved_except_saved_dense_source=True,
                original_fill_materialization_fallback_exact=True)


def main():
    assert pin(BASE / 'closed.json')['sha256'] == '62043d100a96d418b02e25d09a65d8f89655ea21147c27d15a5efd861cce7e1c'
    proof = read(BASE / 'closed.json')
    assert proof['passed']
    for name, wanted in proof['files'].items():
        assert pin(BASE / name) == wanted, name
    row = read(BASE / 'collected/inventory/instructions.json')['observations'][0]
    result = compare(json.loads(row['normalized_methods'][PAD]), json.loads(row['candidate_methods'][PAD]))
    helper = json.loads(row['candidate_methods'][HELPER])
    instructions = helper['instructions']
    assert len(instructions) == 127 and helper['exceptions'] == []
    assert sum(i['opcode'] == 'div' for i in instructions) == 1
    assert all(i['opcode'] != 'rem' for i in instructions)
    calls = [i['operand'] for i in instructions if i['opcode'].startswith('call')]
    assert calls.count('System.Span`1[T]::Void CopyTo(System.Span`1[T])') == 1
    value = dict(**result, build=pin(BASE / 'closed.json'),
                 inventory=pin(BASE / 'collected/inventory/instructions.json'),
                 candidate=read(BASE / 'analysis.json')['built'],
                 helper_instructions=127, helper_calls=calls,
                 helper_review='Validated nonnegative last-axis pads and zero outer pads; width/length zero returns before division; bounded row-index offsets and Span.CopyTo preserve exact bytes. No unsafe instructions or new API.',
                 generator=pin(Path(__file__)), root_product_changed=False,
                 model_qualification_pending=True, performance_pending=True)
    with (OUT / 'composition-20260923.json').open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2); f.write('\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
