"""Inspect actual optimized product code; no component prototype is accepted."""
import hashlib,re

def inspect(text):
    sections=re.split(r'(?=; Assembly listing for method )',text)
    versions=[s for s in sections if s.startswith('; Assembly listing for method ')
              and 'Lokad.Onnx.Tensor' in s.splitlines()[0] and ':LayerNormFloatInto(' in s.splitlines()[0]]
    optimized=[s for s in versions if re.search(r'\((?:FullOpts|Tier1[^)]*)\)',s.splitlines()[0])]
    assert optimized,'No optimized actual product kernel'
    body=optimized[-1]
    for opcode in ['vsubpd','vmulpd','vaddpd','vcvtps2pd']:
        assert re.search(r'\b'+opcode+r'\s+zmm\d+\s*,',body),('Missing wide transform',opcode)
    for opcode in ['vsubpd','vmulpd','vaddpd']:
        assert re.search(r'\b'+opcode+r'\s+ymm\d+\s*,',body),('Missing original vector arithmetic',opcode)
    # .NET 10's disassembler prints both vcvtpd2ps operands with idOpSize.
    assert re.search(r'\bvcvtpd2ps\s+(?:ymm|zmm)\d+\s*,\s*zmm(?:\d+\b|word ptr)',body)
    assert not re.search(r'\bv(?:fma|fnma|fms|fnms)\w*',body),'Unexpected fused arithmetic'
    return dict(passed=True,header=body.splitlines()[0],versions=len(versions),
                optimized_sha256=hashlib.sha256(body.encode()).hexdigest(),
                code_bytes=int(re.search(r'Total bytes of code (\d+)',body)[1]))
