"""Compare complete wrapper IL after removing only constant profiler markers.

Branch and exception destinations are compared as instruction indices because
added markers change byte offsets. No arithmetic, loads, stores, calls, locals
or exception regions may otherwise change. This is for the seven instrumented
tensor wrappers; the two profiler methods require their separate explicit review.
"""
import bisect
import json

PROFILER_CALL = 'Lokad.Onnx.Profiler::Void StartOpStage(Lokad.Onnx.OpStage)'
BRANCHES = {
    'br', 'brfalse', 'brtrue', 'beq', 'bge', 'bgt', 'ble', 'blt',
    'bne.un', 'bge.un', 'bgt.un', 'ble.un', 'blt.un', 'leave',
}
STAGES = set(range(9)) | set(range(1000, 1007))


def constant(instruction):
    opcode, operand = instruction['opcode'], instruction['operand']
    if opcode == 'ldc.i4.m1':
        assert operand == ''
        return -1
    if opcode.startswith('ldc.i4.') and opcode[-1:] in '012345678' and len(opcode) == 8:
        assert operand == ''
        return int(opcode[-1])
    assert opcode in ['ldc.i4', 'ldc.i4.s'], 'Profiler argument is not a constant'
    data = bytes.fromhex(operand)
    assert len(data) == (1 if opcode.endswith('.s') else 4)
    return int.from_bytes(data, 'little', signed=True)


def normalize(body):
    if isinstance(body, str):
        body = json.loads(body)
    instructions = body['instructions']
    assert instructions and instructions[0]['offset'] == 0
    offsets = [i['offset'] for i in instructions]
    assert offsets == sorted(set(offsets)), 'Ambiguous instruction offsets'
    assert instructions[-1]['opcode'] in ['ret', 'throw', 'endfinally', 'rethrow']
    end = offsets[-1] + (2 if instructions[-1]['opcode'] == 'rethrow' else 1)
    boundaries = {*offsets, end}
    removed = set()
    call_offsets = set()
    markers = []
    for index, instruction in enumerate(instructions):
        if instruction['opcode'] == 'call' and instruction['operand'] == PROFILER_CALL:
            assert index > 0 and index - 1 not in removed
            stage = constant(instructions[index - 1])
            assert stage in STAGES, ('Unknown diagnostic stage', stage)
            removed.update([index - 1, index])
            call_offsets.add(instruction['offset'])
            markers.append(stage)
    kept = [i for index, i in enumerate(instructions) if index not in removed]
    assert kept
    kept_offsets = [i['offset'] for i in kept]

    def destination(offset):
        assert offset in boundaries, ('Target is not an instruction boundary', offset)
        assert offset not in call_offsets, 'Control flow enters a profiler call without its constant'
        return bisect.bisect_left(kept_offsets, offset)

    normalized = []
    for index, instruction in enumerate(instructions):
        if index in removed:
            continue
        op, operand = instruction['opcode'], instruction['operand']
        next_offset = instructions[index + 1]['offset'] if index + 1 < len(instructions) else end
        branch = op.removesuffix('.s')
        if branch in BRANCHES:
            raw = bytes.fromhex(operand)
            assert len(raw) == (1 if op.endswith('.s') else 4)
            target = next_offset + int.from_bytes(raw, 'little', signed=True)
            operand = destination(target)
            assert target != end, 'Branch to end of body'
            op = branch
        elif op == 'switch':
            raw = bytes.fromhex(operand)
            assert len(raw) % 4 == 0
            targets = [next_offset + int.from_bytes(raw[i:i + 4], 'little', signed=True)
                       for i in range(0, len(raw), 4)]
            assert all(t != end for t in targets)
            operand = [destination(t) for t in targets]
        normalized.append(dict(opcode=op, operand=operand))

    exceptions = []
    for item in body['exceptions']:
        assert item['TryLength'] > 0 and item['HandlerLength'] > 0
        exceptions.append(dict(flags=item['flags'],
            try_start=destination(item['TryOffset']),
            try_end=destination(item['TryOffset'] + item['TryLength']),
            handler_start=destination(item['HandlerOffset']),
            handler_end=destination(item['HandlerOffset'] + item['HandlerLength']),
            filter=-1 if item['filter'] == -1 else destination(item['filter']), caught=item['caught']))
    return dict(InitLocals=body['InitLocals'], MaxStackSize=body['MaxStackSize'],
                locals=body['locals'], exceptions=exceptions, instructions=normalized), markers


def same_except_markers(before, after):
    original, old_markers = normalize(before)
    diagnostic, new_markers = normalize(after)
    assert original == diagnostic, 'Non-observer IL, locals, stack or control flow changed'
    # Preserve every original label in order. Insertions may return to Math.
    iterator = iter(new_markers)
    assert all(any(actual == expected for actual in iterator) for expected in old_markers), 'Original stages removed or reordered'
    assert len(new_markers) > len(old_markers), 'Expected new timing boundaries'
    return dict(passed=True, original_markers=old_markers, diagnostic_markers=new_markers,
                retained_instructions=len(original['instructions']),
                exception_regions=len(original['exceptions']))
