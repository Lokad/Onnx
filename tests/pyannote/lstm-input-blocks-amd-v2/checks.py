"""Complete census checks shared by the owner and independent local auditor."""
import math
import xml.etree.ElementTree as ET
from protocol import pin, read


def check_suite(path, reference):
    def rows(p):
        doc=ET.parse(p);items=doc.findall('.//{*}UnitTestResult');counts=doc.find('.//{*}Counters').attrib
        assert len(items)==int(counts['total'])==int(counts['passed'])==150 and int(counts['failed'])==0
        assert all(r.attrib['outcome']=='Passed' for r in items)
        assert sum('LstmInputBlockTests.' in r.attrib['testName'] for r in items)==36
        return sorted(r.attrib['testName'] for r in items)
    assert rows(path)==rows(reference)
    return dict(passed=True,tests=150,new_tests=36,trx=pin(path))


def check_result(v,role,width,spec,base,capture=None,native=None):
    if capture is None:capture=read(base/'fixtures/output/result.json')
    if native is None:native=read(base/'fixtures/native/result.json')
    assert v['passed'] and v['role']==role and v['width']==width
    assert v['core']==spec['cores'][role]['sha256'] and v['executable']==spec['consumer']['sha256']
    assert v['calls']==24 and v['distinct_calls']==12 and v['outputs']==72 and v['values']==3631104
    assert v['readonly_operands'] and v['held_outputs_unchanged'] and v['no_performance_measurement']
    assert v['avx512']==(width=='512') and v['hardware_accelerated']==(width!='scalar')
    assert v['flags']==(['DOTNET_EnableAVX512'] if width=='256' else ['DOTNET_EnableHWIntrinsic'] if width=='scalar' else [])
    # System.Numerics.Vector width is observed, not inferred from Avx512F support.
    assert v['vector_count'] in ([4] if width=='scalar' else [8] if width=='256' else [8,16])
    assert v['maximum']==native['maximum']<=1e-4 and len(v['observations'])==72
    for i,row in enumerate(v['observations']):
        call=capture['calls'][i//6];slot=i%3;repeat=i//3%2;reference=native['reports'][i//6*3+slot]
        assert (row['name'],row['index'],row['repeat'],row['slot'])==(call['name'],call['index'],repeat,slot)
        assert row['exact'] and row['sha256']==call['outputs'][slot]['sha256'] and row['values']==call['outputs'][slot]['values']
        assert row['native_maximum']==reference['comparison']['maximum']<=1e-4
        scratch=0 if width=='scalar' else call['inputs'][1]['bytes']+call['inputs'][2]['bytes']+(8192 if role=='candidate' else 0)
        assert row['scratch_bytes']==scratch
    return {k:v[k] for k in ['passed','calls','outputs','values','maximum','vector_count','avx512']}


def check_native(value,output,base):
    import numpy as np
    calls=read(base/'fixtures/output/result.json')['calls']
    assert value['passed'] and value['version']=='1.29.0' and value['no_performance_measurement'] and len(value['reports'])==36
    maximum=0.;values=0
    for i,row in enumerate(value['reports']):
        call=calls[i//3];item=call['outputs'][i%3]
        assert (row['name'],row['index'],row['slot'])==(call['name'],call['index'],i%3)
        assert row['exact_repeat'] and row['input_unchanged'] and row['shape']==item['shape']
        assert pin(output/row['file'])==row['pin'] and row['pin']['bytes']==math.prod(row['shape'])*4
        a=np.fromfile(base/'fixtures/output'/item['file'],dtype='<f4').astype('float64')
        b=np.fromfile(output/row['file'],dtype='<f4').astype('float64')
        assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        errors=np.abs(a-b)/np.maximum(1.,np.abs(b));error=float(errors.max(initial=0))
        assert error==row['maximum']<=1e-4 and row['values']==a.size
        maximum=max(maximum,error);values+=int(a.size)
    assert value['maximum']==maximum and value['values']==values==1815552
    return dict(passed=True,outputs=36,values=values,maximum=maximum,version=value['version'])
