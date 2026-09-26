"""Check the arithmetic actually emitted during focused numerical qualification."""
import hashlib
import re

METHOD = re.compile(r'^; Assembly listing for method ([^\n]+) \(([^\n]+)\)\n(.*?); Total bytes of code (\d+)[^\n]*', re.M | re.S)
INSTRUCTION = re.compile(r'^\s+([0-9A-F]{2,30})\s+(.+)$', re.M)


def inspect(path, mode):
    text = path.read_text(encoding='utf8')
    matches = list(METHOD.finditer(text))
    assert matches and len(matches) == text.count('; Assembly listing for method ') == text.count('; Total bytes of code ')
    listings = []
    loops = []
    for match in matches:
        assert any('CPUExecutionProvider:' + name + '(' in match[1] for name in ['Sigmoid', 'SigmoidRationalVector'])
        ins = [m[2].strip() for m in INSTRUCTION.finditer(match[3])]
        assert ins
        listings.append(dict(signature=match[1], tier=match[2], native_bytes=int(match[4]),
            listing_sha256=hashlib.sha256(match[0].encode()).hexdigest(), instructions=ins))
        if 'SigmoidRationalVector(' not in match[1] or not match[2].startswith('Tier1'): continue
        blocks = re.split(r'(?=^G_M\d+_IG\d+:)', match[3], flags=re.M)
        for block in blocks:
            code = [m[2].strip() for m in INSTRUCTION.finditer(block)]
            if not any(s.startswith('vfmadd') for s in code): continue
            assert sum(s.startswith('vfmadd') for s in code) == 9
            assert any('ymm' in s for s in code) and not any('zmm' in s for s in code)
            assert not any(s.startswith(('call ', 'vcvt')) for s in code)
            assert any(s.startswith('vdivps ') for s in code)
            loops.append(dict(tier=match[2], assembly=block, fmas=9, vector_bits=256,
                calls=[], conversions=[], instructions=len(code)))
    assert any('CPUExecutionProvider:Sigmoid(' in m[1] for m in matches)
    if mode == 'normal':
        assert loops, 'No optimized rational vector body was emitted'
        assert any('SigmoidRationalVector(' in s for r in listings
            if 'CPUExecutionProvider:Sigmoid(' in r['signature'] for s in r['instructions']), 'Helper call missing'
    else:
        assert mode == 'scalar' and not any('SigmoidRationalVector(' in m[1] for m in matches)
    return dict(raw_bytes=path.stat().st_size, raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        listings=listings, rational_loops=loops, no_tier_to_clock_claim=True)
