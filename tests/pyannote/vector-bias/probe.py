"""Replay every captured graph call with new Core and unchanged Data/fixtures."""
from common import *
from phase_audit import audit_preparation

def main():
    assert not (BASE/'probe-processes.json').exists()
    prepared=read(BASE/'prepared.json');preparation=audit_preparation(BASE,sys.modules['common'])
    save(BASE/'preparation-audit.json',preparation)
    capture=ROOT/'artifacts/pyannote-context-reuse-probe-20260921'
    receipt=capture/'closed.json';assert pin(receipt)['sha256']=='d06bffe6d4aba50e7b17373c2a155234c514419bd4859063938fb1fe436fb626'
    reference=read(receipt);assert reference['passed'];verify(reference['files'])
    original=ROOT/'tests/pyannote/context-reuse-probe';source=BASE/'probe-source';source.mkdir()
    shutil.copy2(original/'Probe.csproj',source/'Probe.csproj')
    program=(original/'Program.cs').read_text(encoding='utf8')
    for old,name in [('469cb2d6a4558d917266434bd1f968c8b9f2762795b963800b2d945083852edd','Lokad.Onnx.dll'),
                     ('e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb','Lokad.Onnx.Data.dll')]:
        assert program.count(old)==1;program=program.replace(old,pin(BASE/'runtime'/name)['sha256'])
    (source/'Program.cs').write_text(program,encoding='utf8');shutil.copy2(capture/'manifest.json',BASE/'manifest.json')
    p=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=p.pid,birth=p.create_time()),runs=[])
    state_path=BASE/'probe-processes.json';save(state_path,state)
    flags=monitor.FLAGS+['-p:NuGetAudit=false','-p:FrozenProductDirectory='+str(BASE/'runtime')]
    try:
        for name,cmd in [('probe-restore',['dotnet','restore',source/'Probe.csproj',*flags,'--source',FEED,'--packages',BASE/'packages']),
            ('build',['dotnet','build',source/'Probe.csproj','-c','Release',*flags,'--no-restore','--disable-build-servers','-o',BASE/'bin'])]:
            monitor.worker(state,state_path,name,cmd,source,[0],10,8,900,True,None)
        for path in (BASE/'runtime').glob('*.dll'):
            target=BASE/'bin'/path.name
            if not target.exists():shutil.copy2(path,target)
            assert pin(target)==pin(path)
        files=dict(prepared['files']);files.update(reference['files'])
        for path in [receipt,BASE/'prepared.json',BASE/'preparation-audit.json',BASE/'instructions.json',BASE/'manifest.json',
                     original/'audit.py',ROOT/'tests/pyannote/convolution-pool/phase_audit.py',*TOOLS.iterdir(),*source.iterdir(),*(BASE/'bin').iterdir()]:
            if path.is_file():files[rel(path)]=pin(path)
        save(BASE/'probe-prepared.json',dict(passed=True,files=files,orders=['forward','reverse'],
            limits=dict(preflight_gib=10,rss_gib=8,seconds=900),scope='Captured model output/ownership qualification; allocation counters descriptive, no latency or ORT ratio.'))
        for order in ['forward','reverse']:
            verify(files)
            monitor.worker(state,state_path,order,['dotnet',BASE/'bin/Probe.dll',ROOT,BASE/'manifest.json',BASE/order,order],ROOT,[0],10,8,900,False,BASE/order)
            result=read(BASE/order/'result.json');assert result['passed'] and len(result['records'])==54
            print(order,'54 captured graph calls pass',flush=True)
        verify(files);state['code']=0
    except BaseException:
        state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(state_path,state)

if __name__=='__main__':
    import sys
    main()
