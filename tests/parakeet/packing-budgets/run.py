"""Six fresh complete inference workers, retaining native numerical verdicts."""
from common import *


def main():
    prepared=read(BASE/'prepared.json');assert prepared['passed'];verify(prepared['files'])
    build=read(BASE/'builds.json');assert build['complete'] and build['code']==0;terminal(build['supervisor'])
    for row in build['runs']:
        for pid,birth in row['members'].items():terminal(dict(pid=int(pid),birth=birth))
    assert not (BASE/'processes.json').exists()
    own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    try:
        for mib in (512,2032):
            runtime=BASE/('runtime-'+str(mib));variant=prepared['variants'][str(mib)]
            for mode in ('trace','public','native'):
                name=f'{mib}-{mode}';output=BASE/name;output.mkdir()
                if mode=='native':command=['dotnet',runtime/'TranscribeReplay.dll',ROOT/'models/parakeet-tdt-0.6b-v3',REFERENCE,output/'result.json']
                else:command=['dotnet',runtime/'Profile.dll',ROOT,MANIFEST,output/'output',mode]
                row=worker(state,BASE/'processes.json',name,command,ROOT,[0,1] if mode=='native' else [0],14,12,1200,False,output)
                result=read(output/('result.json' if mode=='native' else 'output/result.json'))
                assert result['core_sha256']==variant['core']['sha256'] and result['data_sha256']==variant['data']['sha256']
                if mode=='native':
                    assert result['application_passed'] and not result['errors'] and result['comparisons']==784 and result['values_compared']==3090494
                    assert row['code']==(0 if result['passed'] else 1);row['native_numeric_passed']=result['passed']
                else:assert result['passed'] and result['inputs_and_held_outputs_unchanged']
                verify(prepared['files']);row['application_passed']=True;save(BASE/'processes.json',state)
                print(name,'application passed; exit',row['code'],flush=True)
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(BASE/'processes.json',state)


if __name__=='__main__':main()
