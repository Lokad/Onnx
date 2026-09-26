"""Retain every targeted listing and code version; make no tier-timing inference."""
import hashlib
import re

METHOD = re.compile(r'^; Assembly listing for method ([^\n]+) \(([^\n]+)\)\n(.*?); Total bytes of code (\d+)[^\n]*', re.M | re.S)
BYTES = re.compile(r'^\s+([0-9A-F]{2,30})\s+(.+)$', re.M)


def inspect(text):
    text = text.replace('\r\n', '\n')
    matches = list(METHOD.finditer(text))
    assert matches and len(matches) == text.count('; Assembly listing for method ') == text.count('; Total bytes of code ')
    rows = []
    for match in matches:
        assert any(n in match[1] for n in ['CPUExecutionProvider:Sigmoid(', 'CPUExecutionProvider:SigmoidRationalVector(']), match[1]
        instructions = []
        for row in BYTES.finditer(match[3]):
            raw = bytes.fromhex(row[1]); assert 0 < len(raw) <= 15
            instructions.append(dict(bytes=row[1], text=row[2].strip()))
        assert instructions, 'Requested code bytes are missing'
        labels = re.findall(r'^(G_M\d+_IG\d+):', match[0], re.M)
        assert labels and len(labels) == len(set(labels))
        assert set(re.findall(r'G_M\d+_IG\d+', match[0])) <= set(labels)
        rows.append(dict(signature=match[1], tier=match[2], native_bytes=int(match[4]),
            listing_sha256=hashlib.sha256(match[0].encode()).hexdigest(), instructions=instructions,
            calls=[r['text'] for r in instructions if re.match(r'^(call|tail\.jmp)\s', r['text'])],
            stack_operands=[r['text'] for r in instructions if re.search(r'\[(?:r|e)(?:sp|bp)[+\-]', r['text'])],
            ymm_instructions=sum(bool(re.search(r'\bymm\d+\b', r['text'])) for r in instructions),
            zmm_instructions=sum(bool(re.search(r'\bzmm\d+\b', r['text'])) for r in instructions)))
    assert any('CPUExecutionProvider:Sigmoid(' in r['signature'] for r in rows)
    return dict(listings=rows, every_emitted_version_retained=True,
                exact_tier_at_each_clock_unknown=True, no_historical_tier_inference=True)
