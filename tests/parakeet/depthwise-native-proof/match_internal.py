"""Add exact same-text PC-relative resolution to the retained original matcher."""
import hashlib
import struct
from match import elf, locate


def resolve_internal(text, offset, kind, symbol, addend, text_index):
    assert kind == 4 and symbol['section'] == text_index and addend == -4
    assert symbol['name'].startswith('MlasConvPostProcessFloatAvx512FFilter')
    assert 0 <= symbol['value'] < len(text)
    struct.pack_into('<i', text, offset, symbol['value']+addend-offset)


def match_object(obj, binary, target):
    sections, symbols, _ = elf(obj); _, _, loads = elf(binary)
    section = sections['.text']; candidate = bytearray(section['data'])
    relocations = []; internal = []
    for s in sections.values():
        if s['type'] != 4 or s['info'] != section['index']: continue
        for p in range(0, s['size'], s['entry']):
            offset, info, addend = struct.unpack_from('<QQq', s['data'], p)
            symbol = symbols[info >> 32]; kind = info & 0xffffffff
            if symbol['section'] == section['index']:
                resolve_internal(candidate, offset, kind, symbol, addend, section['index'])
                internal.append(dict(offset=offset, type=kind, symbol=symbol['name'],
                                     target_offset=symbol['value'], addend=addend))
            else:
                assert symbol['section'] == 0
                relocations.append(dict(offset=offset, type=kind, symbol=symbol['name'], addend=addend))
    matched = locate(candidate, binary, relocations, loads)
    matched['internal_relocations'] = internal
    entries = sorted((s for s in symbols if s['section'] == section['index'] and s['info'] & 15 == 2),
                     key=lambda s:s['value'])
    entry = next(s for s in entries if s['name'] == target)
    end = next((s['value'] for s in entries if s['value'] > entry['value']), section['size'])
    start = matched['start']+entry['value']; end += matched['start']; assert start < end
    return dict(target=target, complete_text=matched, text_bytes=section['size'], symbols=entries,
                start=start, end=end, bytes=end-start, sha256=hashlib.sha256(binary[start:end]).hexdigest(),
                boundary='entry through local helpers/alignment up to next public function or text end; shared epilogues excluded')
