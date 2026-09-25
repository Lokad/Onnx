"""Retained-instruction branch/exception normalization from the layout diagnostic."""
BRANCHES = {'br','brfalse','brtrue','beq','bge','bgt','ble','blt','bne.un','bge.un','bgt.un','ble.un','blt.un','leave'}


def normalized_body(body, retained=None):
    """Express branches and exception regions as retained instruction ordinals."""
    rows = body['instructions']
    if retained is None: retained = list(range(len(rows)))
    assert retained == sorted(set(retained)) and retained[-1] == len(rows)-1
    index_at = {r['offset']:i for i,r in enumerate(rows)}
    # Main ends in ret. No branch into an instruction's operand is acceptable.
    assert rows[-1]['opcode'] == 'ret'
    index_at[rows[-1]['offset']+1] = len(rows)
    ordinals = {}; cursor = len(retained)
    for i in range(len(rows),-1,-1):
        if cursor and retained[cursor-1] == i: cursor -= 1
        ordinals[i] = cursor
    def position(offset): return ordinals[index_at[offset]]
    normalized = []
    for index in retained:
        row = rows[index]; op = row['opcode']; operand = row['operand']
        base = op.removesuffix('.s')
        if base in BRANCHES:
            width = 1 if op.endswith('.s') else 4
            raw = bytes.fromhex(operand); assert len(raw) == width
            target = row['offset']+1+width+int.from_bytes(raw,'little',signed=True)
            operand = position(target); op = base
        elif op == 'switch':
            raw = bytes.fromhex(operand); assert len(raw)%4 == 0
            end = row['offset']+5+len(raw)
            operand = [position(end+int.from_bytes(raw[i:i+4],'little',signed=True)) for i in range(0,len(raw),4)]
        normalized.append(dict(opcode=op,operand=operand))
    exceptions = []
    for item in body['exceptions']:
        value = {k:v for k,v in item.items() if k not in ['TryOffset','TryLength','HandlerOffset','HandlerLength','filter']}
        value.update(try_start=position(item['TryOffset']),try_end=position(item['TryOffset']+item['TryLength']),
            handler_start=position(item['HandlerOffset']),handler_end=position(item['HandlerOffset']+item['HandlerLength']),
            filter=-1 if item['filter'] == -1 else position(item['filter']))
        exceptions.append(value)
    return dict(instructions=normalized,exceptions=exceptions,
        **{k:v for k,v in body.items() if k not in ['instructions','exceptions']})
