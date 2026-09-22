"""Finish unchanged v2 binaries after preserving the 150-versus-149 census failure."""
import importlib.util
from pathlib import Path
import shutil
import sys
import traceback
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/pyannote-lstm-input-blocks-v4-20260922'
PREVIOUS=ROOT/'artifacts/pyannote-lstm-input-blocks-v2-20260922'
sys.path.insert(0,str(ROOT/'tests/pyannote/lstm-input-blocks-v2'))
spec=importlib.util.spec_from_file_location('lstm_blocks_v2',ROOT/'tests/pyannote/lstm-input-blocks-v2/build.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
old.BASE=BASE;old.monitor.BASE=BASE
monitor,pin,read,save,verify=old.monitor,old.pin,old.read,old.save,old.verify
FIXTURES,CORE,CONTROL=old.FIXTURES,old.CORE,old.CONTROL
inventory,models=old.inventory,old.models
FAILED=ROOT/'artifacts/pyannote-lstm-input-blocks-v3-20260922'
JOBS={role+'-'+width:[12,8,900,True] for role in ['selected','candidate'] for width in ['256','scalar']}


def suite(name):
    path=BASE/'test-results'/(name+'.trx');doc=ET.parse(path);rows=doc.findall('.//{*}UnitTestResult');counts=doc.find('.//{*}Counters').attrib
    assert len(rows)==int(counts['total'])==150
    failures=[r for r in rows if r.attrib['outcome']!='Passed']
    assert len(failures)==int(counts['failed'])==(18 if name=='lstm-scalar' else 0)
    assert int(counts['passed'])==(132 if name=='lstm-scalar' else 150)
    assert sum('LstmInputBlockTests.' in r.attrib['testName'] and r.attrib['outcome']=='Passed' for r in rows)==36
    for row in failures:
        assert 'LstmReferenceTests.MultipleBatchesDirectionsActivationsAndStorageMatchOrt(' in row.attrib['testName'] and row.attrib['testName'].endswith('mode: 2)')
        assert row.find('.//{*}Message').text=='System.InvalidOperationException : Tensor intrinsics were explicitly requested but x86 FMA is not supported on this machine.'
        assert 'TensorExecutionOptions.Validate()' in row.find('.//{*}StackTrace').text
    if failures:assert sorted(r.attrib['testName'] for r in failures)==read(FAILED/'failure-closed.json')['unsupported_cases']
    return dict(qualified_supported_cases=True,tests=150,passed=int(counts['passed']),unsupported_explicit_intrinsics=len(failures),new_passed=36,trx=pin(path))


def previous():
    assert pin(PREVIOUS/'failure-closed.json')['sha256']=='f375ecbf942d47e429f0fad51dbd02e32ed59d5800a27e904dcd13786518acde'
    proof=read(PREVIOUS/'failure-closed.json')
    assert proof['closed_failure'] and proof['product_inventory_passed'] and proof['ordinary_tests_passed']==150
    for name,wanted in proof['files'].items():assert pin(PREVIOUS/name)==wanted,name
    for identity in proof['identities']:monitor.terminal(identity)
    for name in ['inputs','binaries']:verify(read(PREVIOUS/(name+'.json'))['files'])
    assert pin(FAILED/'failure-closed.json')['sha256']=='11c93f440a2be617083708421bd3af1239e91316acd04f059164514dfb40fd99'
    failed=read(FAILED/'failure-closed.json');assert failed['closed_failure'] and failed['supported_passed']==132 and failed['new_passed']==36
    for name,wanted in failed['files'].items():assert pin(FAILED/name)==wanted,name
    for identity in failed['identities']:monitor.terminal(identity)
    return proof


def main():
    assert not BASE.exists();proof=previous();BASE.mkdir()
    for folder in ['logs','output','test-results']:(BASE/folder).mkdir()
    for folder in ['runtime','selected-runtime']:shutil.copytree(PREVIOUS/folder,BASE/folder)
    for name in ['instructions.json','instruction-review.json']:shutil.copy2(PREVIOUS/name,BASE/name)
    shutil.copy2(PREVIOUS/'test-results/lstm-ordinary.trx',BASE/'test-results/lstm-ordinary.trx')
    shutil.copy2(FAILED/'test-results/lstm-scalar.trx',BASE/'test-results/lstm-scalar.trx')
    shutil.copy2(ROOT/'PLAN.md',BASE/'prospective-plan.md')
    files={p.as_posix():pin(p) for folder in [BASE,TOOLS] for p in folder.rglob('*') if p.is_file()}
    files.update({(PREVIOUS/name).as_posix():wanted for name,wanted in proof['files'].items()})
    files[(PREVIOUS/'failure-closed.json').as_posix()]=pin(PREVIOUS/'failure-closed.json')
    files.update(read(PREVIOUS/'inputs.json')['files'])
    files[(FAILED/'failure-closed.json').as_posix()]=pin(FAILED/'failure-closed.json')
    files.update({(FAILED/name).as_posix():wanted for name,wanted in read(FAILED/'failure-closed.json')['files'].items()})
    save(BASE/'inputs.json',dict(files=files,no_rebuild=True,previous_failure=pin(PREVIOUS/'failure-closed.json'),jobs=JOBS))
    save(BASE/'binaries.json',dict(files={p.as_posix():pin(p) for folder in [BASE/'runtime',BASE/'selected-runtime'] for p in folder.rglob('*') if p.is_file()}))
    own=monitor.psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    path=BASE/'controller.json';save(path,state)
    def run(name,args,scalar=False,children=False):
        original=monitor.clean_env
        if scalar:monitor.clean_env=lambda:original()|{'DOTNET_EnableHWIntrinsic':'0'}
        try:monitor.worker(state,path,name,args,ROOT,[0],12,8,900,children,BASE/'test-results' if children else BASE/'output')
        finally:monitor.clean_env=original
        print(name,'passed',flush=True)
    try:
        tests={name:suite(name) for name in ['lstm-ordinary','lstm-scalar']}
        for role in ['selected','candidate']:
            folder=BASE/('selected-runtime' if role=='selected' else 'runtime')
            for width in ['256','scalar']:
                run(role+'-'+width,['dotnet',folder/'LstmModelReplay.dll',FIXTURES,BASE/'output'/(role+'-'+width+'.json'),pin(folder/'Lokad.Onnx.dll')['sha256'],role,width],width=='scalar')
        verify(files);verify(read(BASE/'binaries.json')['files'])
        save(BASE/'verified.json',dict(passed=True,core=pin(BASE/'runtime/Lokad.Onnx.dll'),data=pin(BASE/'runtime/Lokad.Onnx.Data.dll'),inventory=inventory(),tests=tests,models=models(),no_performance_measurement=True,actual_amd_pending=True))
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(path,state)


if __name__=='__main__':main()
