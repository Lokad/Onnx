"""Independently reconcile normal product builds, suites, package, inputs and resources."""
import json
from pathlib import Path
import tarfile
from run import BASE,ROOT,prepared
from protocol import JOBS,LIMITS,check_sample,pin,read,save
from checks import inventory,suite,package,consumer,suite256
from warning_census import compare as compare_warnings


def main():
    spec=prepared();assert not (BASE/'closed.json').exists()
    collected=BASE/'collected';receipt=read(collected/'collection.json');transfer=read(BASE/'collection-transfer.json');payload=read(BASE/'payload.json')
    assert transfer['passed'] and transfer['archive']==pin(BASE/'results.tar.gz') and transfer['receipt']==pin(collected/'collection.json')
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None and receipt['payload']==pin(BASE/'payload.json')
    for name,wanted in receipt['files'].items():assert pin(collected/name)==wanted,name
    for name,wanted in payload['files'].items():
        if name.startswith('source/'):
            assert pin(BASE/'bundle'/name)==pin(ROOT/name.removeprefix('source/'))==wanted,name
    with tarfile.open(collected/'evidence/tensor-source.tar') as tar:
        import hashlib
        for member in tar.getmembers():
            if member.isfile():
                data=tar.extractfile(member).read();assert payload['files']['source/'+member.name]==dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    state=read(collected/'identity.json');assert state['complete'] and state['code']==0 and state['supervisor']==read(BASE/'deployment.json')
    assert [r['name'] for r in state['runs']]==payload['jobs']==JOBS and state['boot_time']==1789634288.0
    assert receipt['identities']==[state['supervisor']]+[dict(pid=int(p),birth=b) for r in state['runs'] for p,b in r['members'].items()]
    resources=[]
    for row in state['runs']:
        assert row['complete'] and row['code']==0 and row['seconds']<900
        assert row['preflight']['available']>=LIMITS['preflight_available'] and row['preflight']['tmpfs']>=LIMITS['preflight_tmpfs']
        samples=[json.loads(s) for s in (collected/'logs'/(row['name']+'.jsonl')).read_text().splitlines()]
        assert len(samples)==row['samples']>0 and max(s['rss'] for s in samples)==row['peak_rss']
        for sample in samples:
            check_sample(sample)
            assert all(row['members'][str(m['pid'])]==m['birth'] for m in sample['members'])
        resources.append(dict(name=row['name'],samples=len(samples),peak_rss=row['peak_rss'],seconds=row['seconds']))
    assert state['ended']-state['started']<4*3600
    assert (collected/'logs/sdk-version.stdout').read_text().strip().endswith('10.0.204')
    built=read(collected/'built.json');assert built['passed']
    for name,wanted in built['files'].items():
        if name.startswith(('runtime/','built/')):assert pin(collected/name)==wanted,name
    for name,wanted in built['product'].items():
        assert pin(collected/'runtime'/name)==wanted
        assert built['files']['source/tests/Lokad.Onnx.Backend.Tests/bin/Release/net10.0/'+name]==wanted
    assert built['files']['source/tests/Lokad.Onnx.Tensors.Tests/bin/Release/net10.0/Lokad.Onnx.dll']==built['product']['Lokad.Onnx.dll']
    il=inventory(read(collected/'inventory/instructions.json'),payload['measured'],built['product'])
    assert il==read(collected/'inventory/review.json')
    suites={}
    for name in ['backend','tensors']:
        suites[name]=suite(collected/(name+'-tests')/(name+'.trx'),name,collected/'evidence')
        assert suites[name]==read(collected/(name+'-tests')/'review.json')
    widths={}
    for name in ['backend','tensors']:
        widths[name]=suite256(collected/(name+'-tests-256')/(name+'.trx'),name,collected/'evidence')
        assert widths[name]==read(collected/(name+'-tests-256')/'review.json')
    pkg=package(collected/'nuget/Lokad.Onnx.0.2.0.nupkg',built['product']['Lokad.Onnx.dll'])
    assert pkg==read(collected/'package/review.json')
    probe=read(collected/'consumer-built.json');assert probe['passed'] and pin(collected/'built/PackageProbe.dll')==probe['executable']
    assert probe['files']['consumer/bin/Release/net10.0/Lokad.Onnx.dll']==built['product']['Lokad.Onnx.dll']
    value=read(collected/'consumer-run/result.json');assert value['pid']==state['runs'][-1]['child']['pid']
    consumed=consumer(value,built['product']['Lokad.Onnx.dll'],probe['executable'],payload['files']['source/tests/Lokad.Onnx.Backend.Tests/models/mnist-8.onnx'])
    assert consumed==read(collected/'consumer-run/review.json')
    assert read(collected/'evidence/root-applied.json')==read(BASE/'bundle/evidence/root-applied.json')
    assert all(payload['files']['source/'+name]==wanted for name,wanted in read(collected/'evidence/root-applied.json')['source_files'].items())
    warnings=compare_warnings(collected)
    analysis=dict(passed=True,warnings=warnings,root_source_verified=True,root_integration=pin(collected/'evidence/root-applied.json'),measured=payload['measured'],built=built['product'],inventory=il,suites=suites,suite256=widths,package=pkg,consumer=consumed,resources=resources,no_performance_measurement=True)
    save(BASE/'analysis.json',analysis)
    files={p.relative_to(BASE).as_posix():pin(p) for p in BASE.rglob('*') if p.is_file()}
    save(BASE/'closed.json',dict(passed=True,files=files,local_inputs=spec['files'],remote_terminal=receipt['identities'],analysis=pin(BASE/'analysis.json')))
    print(json.dumps(dict(closed=pin(BASE/'closed.json'),**analysis)))


if __name__=='__main__':main()
