import traceback
from common import *


def main():
    prepared=read(BASE/'prepared.json');assert prepared['passed'];verify(prepared['files'])
    for name,wanted in prepared['external_files'].items():assert pin(Path(name))==wanted,name
    assert not (BASE/'processes.json').exists()
    own=psutil.Process();state=dict(complete=False,code=None,supervisor=dict(pid=own.pid,birth=own.create_time()),runs=[])
    try:
        for index,role in enumerate(prepared['jobs']):
            name=f'{index}-{role}';folder=BASE/name;folder.mkdir()
            prefix=[sys.executable,'-X','utf8','-B',NATIVE] if role=='ort' else ['dotnet',BASE/('runtime-'+role)/'AudioBenchmark.dll']
            command=prefix+[ROOT,INPUT,folder/'output','timing']
            row=monitor.worker(state,BASE/'processes.json',name,command,ROOT,[0],14,12,1800,False,folder)
            result=read(folder/'output/result.json');assert len(result['records'])==80 and result['held_outputs_unchanged']
            if role!='ort':
                assert result['core_sha256']==prepared['roles'][role]['Lokad.Onnx.dll']['sha256']
                assert result['data_sha256']==prepared['roles'][role]['Lokad.Onnx.Data.dll']['sha256']
            else:assert result['native_binaries']==prepared['native_binaries'] and result['native_settings']==prepared['native_settings']
            verify(prepared['files']);row['application_passed']=True;save(BASE/'processes.json',state)
            print(name,'80 requests passed',flush=True)
        state['code']=0
    except BaseException:state.update(code=1,error=traceback.format_exc());raise
    finally:state['complete']=True;save(BASE/'processes.json',state)


if __name__=='__main__':main()
