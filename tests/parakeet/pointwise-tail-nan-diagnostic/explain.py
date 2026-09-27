"""Reconstruct every first mismatch using the observed multiply operand order."""
import json
import re
import struct
from run import BASE,FAILED,pin,read,write,prepared


BITS=[0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc12345,0xffc54321,0x7fa12345]
def as_float(value):return struct.unpack('<f',struct.pack('<I',value))[0]
def as_bits(value):
    try:return struct.unpack('<I',struct.pack('<f',value))[0]
    except OverflowError:return 0x7f800000 if value>0 else 0xff800000
def is_nan(value):return (value&0x7fffffff)>0x7f800000
def operation(a,b,multiply):
    if is_nan(a):return a|0x400000
    if is_nan(b):return b|0x400000
    result=as_bits(as_float(a)*as_float(b) if multiply else as_float(a)+as_float(b))
    return 0xffc00000 if is_nan(result) else result
def values(count,seed):
    result=[]
    for i in range(count):
        seed=(seed*1664525+1013904223)&0xffffffff
        result.append(BITS[(i+seed%9)%9] if i%17<9 else as_bits(((seed>>8)%2001-1000)/997.0))
    return result


def main():
    prepared();review=read(BASE/'review.json')
    assert review['diagnostic_complete'] and review['products_unchanged'] and not review['case_differences']
    path=BASE/'capture-collected/logs/avx512-disabled.stdout'
    assert pin(path)==review['disassembly']
    blocks=path.read_text().split('; Assembly listing for method ')[1:]
    helper=next(b for b in blocks if 'PackedColumnMaskedEightRows' in b.splitlines()[0])
    assert helper.count('vmulps')==8 and helper.count('vaddps')==8
    assert helper.count('vmulps   ymm10, ymm10, ymm9')==7 and 'vmulps   ymm9, ymm10, ymm9' in helper
    assert 'vpmaskmovd ymm9, ymm0,' in helper and 'vbroadcastss ymm10,' in helper
    baseline=[b for b in blocks if '(Tier1)' in b.splitlines()[0] and ':PackedColumn' not in b]
    assert len(baseline)==1 and 'vmulps   ymm4, ymm3, ymm4' in baseline[0] and 'vmulps   ymm3, ymm3, ymm5' in baseline[0]
    old=read(FAILED/'capture-collected/probe/avx512-disabled/result.json')
    normal=read(FAILED/'capture-collected/probe/normal/result.json')
    explained=[];cross=[]
    for case in old['results']:
        if case['bit_exact']:continue
        m,n,k=(case[key] for key in ['m','n','k'])
        assert case['exceptional']
        hit=re.search(r'index=(\d+) expected=([0-9a-f]+) actual=([0-9a-f]+)',case['error'])
        index=int(hit[1])-7;row,col=divmod(index,k)
        assert col>=k-k%8 and ',True,0 index=' in case['error']
        a=values((row+1)*n,17);b=values(n*k,31);expected=actual=values(index+1,53)[index]
        for j in range(n):
            av,bv=a[row*n+j],b[j*k+col]
            expected=operation(operation(bv,av,True),expected,False)
            actual=operation(operation(av,bv,True),actual,False)
        assert (expected,actual)==(int(hit[2],16),int(hit[3],16))
        explained.append(dict(m=m,n=n,k=k,row=row,column=col,expected=f'{expected:08x}',actual=f'{actual:08x}'))
    assert len(explained)==260
    for a,b in zip(normal['results'],old['results'],strict=True):
        if a['bit_exact'] and b['bit_exact'] and a['output_sha256']!=b['output_sha256']:
            assert a['exceptional'];cross.append({k:a[k] for k in ['m','n','k','exceptional']})
    assert len(cross)==482 and sum(c['m']<64 or c['n']<64 for c in cross)==434
    write(BASE/'explanation.json',dict(diagnostic_complete=True,review=pin(BASE/'review.json'),explainer=pin(__import__('pathlib').Path(__file__)),
        arithmetic_inference='Observed baseline B*A then product+C; candidate A*B then product+C. Quiet the first NaN operand, otherwise perform the separate float32 operations.',
        reproduced_first_mismatches=explained,all_reproduced=True,cross_mode_exceptional_differences=cross,
        no_performance_measurement=True,release_admitted=False))
    print(json.dumps(dict(explanation=pin(BASE/'explanation.json'),reproduced=len(explained),cross_mode_differences=len(cross))))


if __name__=='__main__':main()
