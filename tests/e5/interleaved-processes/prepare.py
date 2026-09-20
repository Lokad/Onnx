"""Freeze both phases after complete local private-process smoke and refusals."""
from pathlib import Path
import argparse,shutil,subprocess,tarfile
from audit import pin,read,write,worker
from protocol import CORE,NATIVE,PROTOCOL,LIMITS,CRITERIA,schedule,ROLES,cycles,commands

ROOT=Path(__file__).resolve().parents[3]

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(),'Commit tools before freezing'
    smoke=read(base/'smoke-audit.json');assert smoke['passed'] is True
    assert 'OK' in (base/'analysis-tests-final.log').read_text()
    assert '0 Warning(s)' in (base/'build.log').read_text() and '0 Error(s)' in (base/'build.log').read_text()
    for name,wanted in smoke['binaries'].items():assert pin(base/'bin'/name)==wanted,name
    for name in ['Program.cs','Probe.csproj','run.py','protocol.py','audit.py']:
        current=Path(__file__).with_name(name)
        if name=='audit.py' and pin(current)!=smoke['source'][name]:
            review=read(base/'whitespace-review.json');original=base/'audit-before-whitespace.py'
            assert pin(original)==review['before']==smoke['source'][name] and pin(current)==review['after']
            assert original.read_bytes().rstrip()==current.read_bytes().rstrip() and review['only_trailing_whitespace'] is True
        else:assert pin(current)==smoke['source'][name],name
    for phase in ['aa','compare']:
        state=read(base/('smoke-'+phase)/'identity.json');assert state['complete'] and state['code']==0
        for job in [j for j in schedule() if j['visit']==0]:
            for role in ROLES:worker(base/('smoke-'+phase)/job['name']/role/'output',base/'inputs',smoke['model'],
                smoke['binaries']['InterleavedProcesses.dll'],smoke['binaries']['Microsoft.ML.OnnxRuntime.dll'],smoke['native'],job|dict(role=role),phase,True)
    assert smoke['binaries']['Lokad.Onnx.dll']['sha256']==CORE
    qualified=ROOT/'artifacts/e5-layernorm-product-20260920/closed.json'
    assert pin(qualified)['sha256']=='5afd88417ad896ef416df9ddfe32d8a3bb61c32332f2cd7010ffe57d1c66b2b8'
    diagnosis=ROOT/'artifacts/e5-deployment-variance-20260920/closed.json'
    assert pin(diagnosis)['sha256']=='c1c16b942b01a405a470e5c036c8df72b25f34aeedb0b5334ca9fb6229436bd8'
    native=ROOT/'artifacts/e5-public-ort-20260919/bin/libonnxruntime.so';assert pin(native)['sha256']==NATIVE
    model=ROOT/'models/multilingual-e5-small/model.onnx';assert pin(model)==smoke['model']
    payload=base/'payload';payload.mkdir();shutil.copytree(base/'bin',payload/'bin');shutil.copytree(base/'inputs',payload/'inputs')
    for path in Path(__file__).parent.iterdir():
        if path.is_file() and path.suffix in ['.cs','.csproj','.py','.md']:shutil.copyfile(path,payload/path.name)
    shutil.copyfile(ROOT/'eng/campaign_processes.py',payload/'campaign_processes.py');shutil.copyfile(ROOT/'global.json',payload/'global.json')
    shutil.copyfile(ROOT/'.agent/m2-interleaved-processes-20260920.md',payload/'prospective-plan.md')
    meta=dict(schema=1,protocol=PROTOCOL,source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        product_source='4f10e8bc70627f33d00ffb06ddccf15f43464d2a',qualified_product_receipt=pin(qualified),diagnostic_receipt=pin(diagnosis),
        model=dict(path='/home/vermorel/Onnx/models/multilingual-e5-small/model.onnx',**pin(model)),
        native=dict(path='/home/vermorel/Onnx/artifacts/e5-public-ort-20260919/worker/libonnxruntime.so',**pin(native)),
        limits=LIMITS,criteria=CRITERIA,schedule=schedule(),cycles={j['name']:cycles(j) for j in schedule()},commands={j['name']:commands(j) for j in schedule()},
        phases=['aa','compare'],smoke_audit=pin(base/'smoke-audit.json'),analysis_tests=pin(base/'analysis-tests-final.log'),
        files={p.relative_to(payload).as_posix():pin(p) for p in sorted(payload.rglob('*')) if p.is_file()})
    write(payload/'frozen.json',meta)
    with tarfile.open(base/'payload.tar.gz','x:gz') as tar:
        for name in list(meta['files'])+['frozen.json']:tar.add(payload/name,arcname=name,recursive=False)
    write(base/'preparation.json',dict(archive=pin(base/'payload.tar.gz'),frozen=pin(payload/'frozen.json')))
    print('Frozen',len(meta['files']),'files;',pin(base/'payload.tar.gz'))

if __name__=='__main__':main()
