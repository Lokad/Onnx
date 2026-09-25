"""Identify complete MLAS object text, resolving rather than ignoring relocations."""
import hashlib
import struct


def elf(data):
    assert data[:6] == b'\x7fELF\x02\x01'
    h = struct.unpack_from('<16sHHIQQQIHHHHHH', data)
    assert h[2] == 62  # AMD64
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, h[6]+i*h[11]) for i in range(h[12])]
    strings = sections[h[13]]
    names = data[strings[4]:strings[4]+strings[5]]
    result = {}
    for index, s in enumerate(sections):
        name = names[s[0]:].split(b'\0', 1)[0].decode()
        result[name] = dict(index=index, type=s[1], address=s[3], offset=s[4], size=s[5],
                            link=s[6], info=s[7], entry=s[9], data=data[s[4]:s[4]+s[5]])
    symbols = []
    if '.symtab' in result:
        s = result['.symtab']; t = sections[s['link']]; strings = data[t[4]:t[4]+t[5]]
        for offset in range(0, s['size'], s['entry']):
            name, info, other, section, value, size = struct.unpack_from('<IBBHQQ', s['data'], offset)
            symbols.append(dict(name=strings[name:].split(b'\0', 1)[0].decode(), info=info,
                                section=section, value=value, size=size))
    loads = []
    for i in range(h[10]):
        p = struct.unpack_from('<IIQQQQQQ', data, h[5]+i*h[9])
        if p[0] == 1:
            loads.append(dict(flags=p[1], offset=p[2], address=p[3], size=p[5]))
    return result, symbols, loads


def address(loads, offset, length=1, executable=False):
    found = [p for p in loads if p['offset'] <= offset and offset+length <= p['offset']+p['size']
             and (not executable or p['flags'] & 1)]
    assert len(found) == 1
    return offset-found[0]['offset']+found[0]['address']


def file_offset(loads, value, length):
    found = [p for p in loads if p['address'] <= value and value+length <= p['address']+p['size']]
    assert len(found) == 1
    return value-found[0]['address']+found[0]['offset']


def locate(candidate, original, relocations, loads):
    excluded = {i for r in relocations for i in range(r['offset'], r['offset']+4)}
    spans = []; start = None
    for i in range(len(candidate)+1):
        if i < len(candidate) and i not in excluded:
            if start is None: start = i
        elif start is not None:
            spans.append((start, i)); start = None
    a, b = max(spans, key=lambda s:s[1]-s[0]); assert b-a >= 64
    matches = []; search = 0
    while True:
        pos = original.find(candidate[a:b], search)
        if pos < 0: break
        search = pos+1; begin = pos-a
        if begin < 0 or begin+len(candidate) > len(original): continue
        if not all(candidate[x:y] == original[begin+x:begin+y] for x,y in spans): continue
        try: native_address = address(loads, begin, len(candidate), executable=True)
        except AssertionError: continue
        fixed = bytearray(candidate); resolved = []
        for r in relocations:
            # Only these two declared PC-relative references are admitted.
            assert r['type'] == 2 and r['symbol'] == 'MlasMaskMoveAvx'
            assert r['addend'] in [-4, 12]
            displacement = struct.unpack_from('<i', original, begin+r['offset'])[0]
            symbol_address = displacement+native_address+r['offset']-r['addend']
            constant_offset = file_offset(loads, symbol_address, 32)
            assert original[constant_offset:constant_offset+32] == struct.pack('<8I', *range(8))
            struct.pack_into('<i', fixed, r['offset'], symbol_address+r['addend']-(native_address+r['offset']))
            resolved.append(dict(**r, symbol_address=hex(symbol_address), constant_offset=hex(constant_offset)))
        if bytes(fixed) == original[begin:begin+len(candidate)]:
            matches.append(dict(start=begin, end=begin+len(candidate), address=native_address,
                                relocations=resolved, sha256=hashlib.sha256(fixed).hexdigest()))
    assert len(matches) == 1, ('complete text matches', len(matches))
    return matches[0]


def match_object(obj, binary, target):
    sections, symbols, _ = elf(obj); _, _, loads = elf(binary)
    section = sections['.text']; relocations = []
    for s in sections.values():
        if s['type'] != 4 or s['info'] != section['index']: continue
        for p in range(0, s['size'], s['entry']):
            offset, info, addend = struct.unpack_from('<QQq', s['data'], p)
            symbol = symbols[info >> 32]
            relocations.append(dict(offset=offset, type=info & 0xffffffff,
                                    symbol=symbol['name'], addend=addend))
    matched = locate(section['data'], binary, relocations, loads)
    entries = sorted((s for s in symbols if s['section'] == section['index'] and s['info'] & 15 == 2),
                     key=lambda s:s['value'])
    entry = next(s for s in entries if s['name'] == target)
    end = next((s['value'] for s in entries if s['value'] > entry['value']), section['size'])
    start = matched['start']+entry['value']; end += matched['start']
    assert start < end
    return dict(target=target, complete_text=matched, text_bytes=section['size'], symbols=entries,
                start=start, end=end, bytes=end-start,
                sha256=hashlib.sha256(binary[start:end]).hexdigest(),
                boundary='entry through private helpers/alignment up to next public function or text end')
