"""Independently reconcile immutable inputs, every test/output, and terminal resources."""
import importlib.util
import json
import tarfile
import xml.etree.ElementTree as ET
from run import ROOT, BASE, PREVIOUS, FIXTURES, TOOLS, CONTROL, CORE, JOBS, pin, read, save, verify, inventory, models, previous
from transform import PANELS, RECURRENT, HELPER, once, transform


def main():
    assert not (BASE/'closed.json').exists()
    value=read(BASE/'verified.json'); assert value['passed'] and value['actual_amd_pending'] and value['no_performance_measurement']
    for name in ['inputs.json','binaries.json']:verify(read(BASE/name)['files'])
    spec=importlib.util.spec_from_file_location('lstm_blocks_resources',ROOT/'tests/parakeet/portable-models/common.py')
    common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
    previous()
    jobs=JOBS
    resources=common.resources(BASE,'controller.json',jobs);state=read(BASE/'controller.json')
    with tarfile.open(PREVIOUS/'source.tar') as archive:
        texts,patch=transform(archive.extractfile(PANELS).read().decode('utf8'),archive.extractfile(RECURRENT).read().decode('utf8'))
        assert (PREVIOUS/'candidate.patch').read_text()==patch
        for path,text in texts.items():assert (PREVIOUS/'source'/path).read_text()==text
        old='tests/Lokad.Onnx.Backend.Tests/LstmOutputLaneTests.cs'
        original=archive.extractfile(old).read().decode('utf8').replace('\r\n','\n')
        assert (PREVIOUS/'scratch-test-before.cs.txt').read_text()==original
        assert (PREVIOUS/'source'/old).read_text()==once(original,'3L * (w.Length + r.Length) * sizeof(float)','3L * (w.Length + r.Length + 16 * hidden) * sizeof(float)')
        # Every other original source/test/metadata file remains byte exact.
        for member in archive.getmembers():
            if member.isfile() and member.name not in [PANELS,RECURRENT,old]:
                assert (PREVIOUS/'source'/member.name).read_bytes()==archive.extractfile(member).read(),member.name
    assert pin(PREVIOUS/'source/tests/Lokad.Onnx.Backend.Tests/LstmInputBlockTests.cs')==pin(ROOT/'tests/pyannote/lstm-input-blocks-v2/LstmInputBlockTests.cs.txt')
    assert value['core']==pin(BASE/'runtime/Lokad.Onnx.dll') and value['data']==pin(BASE/'runtime/Lokad.Onnx.Data.dll')
    assert pin(BASE/'selected-runtime/Lokad.Onnx.dll')['sha256']==CORE
    assert inventory()==value['inventory']==read(BASE/'instruction-review.json')
    prior=read(PREVIOUS/'controller.json');assert prior['complete'] and prior['code']==1
    # Audit the preceding successful workers without changing the preserved failed controller.
    common_spec=read(BASE/'inputs.json');assert common_spec['no_rebuild']
    for row in prior['runs']:
        assert row['complete'] and row['code']==0
        samples=[json.loads(v) for v in (PREVIOUS/'logs'/(row['name']+'.samples.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples'] and max(v['rss'] for v in samples)==row['peak_rss']
        for v in samples:
            assert v['rss']<8*1024**3 and v['seconds']<900 and v['available']>=1024**3 and v['disk']>=20*1024**3 and v['output_bytes']<=1024**3
            assert v['rss']==sum(p['rss'] for p in v['members']) and all(p['affinity']==[2] and row['members'][str(p['pid'])]==p['birth'] for p in v['members'])
    test_names=[]
    for mode in ['ordinary','scalar']:
        name='lstm-'+mode;path=BASE/'test-results'/(name+'.trx');doc=ET.parse(path)
        rows=doc.findall('.//{*}UnitTestResult');counter=doc.find('.//{*}Counters').attrib
        assert int(counter['total'])==int(counter['passed'])==150 and int(counter['failed'])==0 and len(rows)==150
        assert all(r.attrib['outcome']=='Passed' for r in rows)
        assert sum('LstmInputBlockTests.' in r.attrib['testName'] for r in rows)==36
        assert value['tests'][name]==dict(passed=True,tests=150,new_tests=36,trx=pin(path))
        test_names.append(sorted(r.attrib['testName'] for r in rows))
    assert test_names[0]==test_names[1]
    assert models()==value['models']
    capture=read(FIXTURES/'output/result.json');native=read(FIXTURES/'native/result.json')
    import numpy as np
    maximum=0.;count=0
    for ordinal,call in enumerate(capture['calls']):
        for slot,item in enumerate(call['outputs']):
            report=native['reports'][ordinal*3+slot];ref=report['reference']
            a=np.fromfile(FIXTURES/'output'/item['file'],dtype='<f4').astype('float64')
            b=np.fromfile(FIXTURES/'native'/ref['file'],dtype='<f4').astype('float64')
            assert a.shape==b.shape and a.size==item['values'] and np.isfinite(a).all() and np.isfinite(b).all()
            error=np.abs(a-b)/np.maximum(1.,np.abs(b));peak=float(error.max(initial=0))
            assert peak==report['comparison']['maximum']<=1e-4 and not (error>1e-4).any()
            maximum=max(maximum,peak);count+=int(a.size)
    assert count==1815552 and maximum==native['maximum']
    for role in ['selected','candidate']:
        folder=BASE/('selected-runtime' if role=='selected' else 'runtime')
        for width in ['256','scalar']:
            name=role+'-'+width;result=read(BASE/'output'/(name+'.json'))
            worker,=[r['worker'] for r in state['runs'] if r['name']==name]
            assert result['pid']==worker['pid'] and result['executable']==pin(folder/'LstmModelReplay.dll')['sha256']
            assert result['maximum']==maximum and result['values']==2*count
    analysis=dict(passed=True,tests_per_mode=150,new_tests_per_mode=36,models=value['models'],native_maximum=maximum,
        core=value['core'],data=value['data'],inventory=value['inventory'],resources=resources['resources'],
        no_performance_measurement=True,actual_amd_pending=True)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file() and not {'obj','packages'}.intersection(p.relative_to(BASE).parts)}
    save(BASE/'closed.json',dict(passed=True,files=files,identities=resources['identities'],local_inputs=read(BASE/'inputs.json')['files']))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__':main()
