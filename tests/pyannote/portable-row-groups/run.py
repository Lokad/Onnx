from common import *

def main():
    spec=read(BASE/'prepared.json');verify(spec['files'])
    path=BASE/'processes.json';assert not path.exists()
    st=state();save(path,st)
    try:
        for name in spec['jobs']:
            mode=name.split('-')[0]
            cmd=['dotnet',BASE/'runtime/RowGroupProbe.dll',BASE/'shapes.json',mode,BASE/'output'/(name+'.json')]
            monitor.worker(st,path,name,cmd,ROOT,[0],8,4,900,False,BASE/'output')
            assert read(BASE/'output'/(name+'.json'))['passed']
        st['code']=0
    except BaseException:
        st.update(code=1,error=traceback.format_exc()); raise
    finally:
        st['complete']=True;save(path,st)

if __name__=='__main__': main()
