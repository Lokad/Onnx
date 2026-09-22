"""Preserve every captured output, native error, mode and ownership check."""
import math
from protocol import pin,read


def check_result(v,role,width,spec,base,capture=None,native=None):
    if capture is None:capture=read(base/'fixtures/output/result.json')
    if native is None:native=read(base/'fixtures/native/result.json')
    assert v['passed'] and v['role']==role and v['width']==width
    assert v['core']==spec['cores'][role]['sha256'] and v['executable']==spec['consumer']['sha256']
    assert v['calls']==24 and v['distinct_calls']==12 and v['outputs']==72 and v['values']==3631104
    assert v['readonly_operands'] and v['held_outputs_unchanged'] and v['no_performance_measurement']
    assert v['avx512']==(width in ['512','simd']) and v['hardware_accelerated']==(width!='scalar')
    assert v['flags']==(['DOTNET_EnableAVX512'] if width=='256' else ['DOTNET_EnableHWIntrinsic'] if width=='scalar' else [])
    # System.Numerics.Vector width is observed, not inferred from Avx512F support.
    assert v['vector_count'] in ([4] if width=='scalar' else [8] if width=='256' else [8,16])
    assert v['maximum']==native['maximum']<=1e-4 and len(v['observations'])==72
    for i,row in enumerate(v['observations']):
        call=capture['calls'][i//6];slot=i%3;repeat=i//3%2;reference=native['reports'][i//6*3+slot]
        assert (row['name'],row['index'],row['repeat'],row['slot'])==(call['name'],call['index'],repeat,slot)
        assert row['exact'] and row['sha256']==call['outputs'][slot]['sha256'] and row['values']==call['outputs'][slot]['values']
        assert row['native_maximum']==reference['comparison']['maximum']<=1e-4
        scratch=0 if width=='scalar' else call['inputs'][1]['bytes']+call['inputs'][2]['bytes']+8192
        assert row['scratch_bytes']==scratch
    return {k:v[k] for k in ['passed','calls','outputs','values','maximum','vector_count','avx512']}
