"""Require observed optimized wide arithmetic, retaining all emitted versions."""
import hashlib,re

def inspect(text):
    sections=re.split(r'(?=; Assembly listing for method )',text)
    def method(name):
        candidates=[s for s in sections if s.startswith('; Assembly listing for method ') and ':'+name+'(' in s.splitlines()[0]]
        optimized=[s for s in candidates if re.search(r'\((?:FullOpts|Tier1[^)]*)\)',s.splitlines()[0])]
        assert optimized,'No optimized body: '+name
        return candidates,optimized[-1]
    product_versions,product=method('LayerNormFloatInto');wide_versions,wide=method('WideOutput')
    assert 'Lokad.Onnx.Tensor' in product.splitlines()[0] and 'LayerNormOutput.Kernels:' in wide.splitlines()[0]
    for opcode in ['vsubpd','vmulpd','vaddpd','vcvtps2pd']:
        assert re.search(r'\b'+opcode+r'\s+zmm\d+\s*,',wide),('Missing wide destination',opcode)
    assert re.search(r'\bvcvtpd2ps\s+ymm\d+\s*,\s*zmm(?:\d+|word ptr)',wide),'Missing narrowing from wide doubles'
    for opcode in ['vsubpd','vmulpd','vaddpd']:
        assert re.search(r'\b'+opcode+r'\s+ymm\d+\s*,',product),('Missing product vector operation',opcode)
        assert not re.search(r'\b'+opcode+r'\s+[^\n;]*\bzmm\d+\b',product),'Unexpected wider product arithmetic'
    assert not re.search(r'\bv(?:fma|fnma|fms|fnms)\w*',product+wide),'Fused arithmetic changes association'
    return dict(passed=True,product_header=product.splitlines()[0],wide_header=wide.splitlines()[0],
        product_versions=len(product_versions),wide_versions=len(wide_versions),
        product_optimized_sha256=hashlib.sha256(product.encode()).hexdigest(),wide_optimized_sha256=hashlib.sha256(wide.encode()).hexdigest(),
        product_code_bytes=int(re.search(r'Total bytes of code (\d+)',product)[1]),wide_code_bytes=int(re.search(r'Total bytes of code (\d+)',wide)[1]))
