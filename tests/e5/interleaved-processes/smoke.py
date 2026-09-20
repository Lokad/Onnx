"""All five cases/policies, both phase settings, four real private processes."""
from pathlib import Path
import argparse,json,shutil,sys,time,traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.append(str(ROOT/'artifacts/asr-labeled-20260919/venv/Lib/site-packages'))
import psutil
from protocol import schedule,LIMITS,ROLES
from run import cohort,save,terminal
from audit import pin,worker,telemetry,write

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);a=p.parse_args();base=a.artifact.resolve()
    base.mkdir(exist_ok=True)
    shutil.copytree(Path(__file__).parent/'bin/Release/net10.0',base/'bin')
    shutil.copytree(ROOT/'artifacts/e5-layernorm-product-20260920/payload/inputs',base/'inputs')
    model=ROOT/'models/multilingual-e5-small/model.onnx';native=ROOT/'artifacts/e5-public-ort-20260919/bin/onnxruntime.dll'
    parent=psutil.Process();old=parent.cpu_affinity();parent.cpu_affinity([0]);audits={}
    source={p.name:pin(p) for p in Path(__file__).parent.iterdir() if p.is_file()}
    binaries={p.name:pin(p) for p in (base/'bin').iterdir() if p.is_file()}
    try:
        for phase in ['aa','compare']:
            out=base/('smoke-'+phase);out.mkdir();jobs=[j for j in schedule() if j['visit']==0]
            state=dict(phase=phase,supervisor=dict(pid=parent.pid,birth=parent.create_time()),started=time.time(),complete=False,limits=LIMITS,runs=[])
            def save_state():save(out/'identity.json',state)
            save_state()
            try:
                for job in jobs:cohort(base,out,job,phase,model,native,True,state,save_state)
                state['complete']=True;state['code']=0
            except BaseException:state['code']=2;state['error']=traceback.format_exc();raise
            finally:state['ended']=time.time();save_state()
            values=[]
            for job in jobs:
                for role in ROLES:
                    values.append(worker(out/job['name']/role/'output',base/'inputs',pin(model),binaries['InterleavedProcesses.dll'],
                        binaries['Microsoft.ML.OnnxRuntime.dll'],pin(native),job|dict(role=role),phase,True))
            records={j['name']:[json.loads(l) for l in (out/j['name']/'samples.jsonl').read_text().splitlines()] for j in jobs}
            resources=telemetry(state,records,jobs,True);terminal(resources['births'][1:])
            audits[phase]=dict(passed=True,resources=resources,outputs=[dict(specification=v['specification'],sha256=v['output_sha256'],error=v['after_error']) for v in values])
    finally:parent.cpu_affinity(old)
    for index in range(5):
        for native_role in [False,True]:
            rows=[v for audit in audits.values() for v in audit['outputs'] if v['specification']['case_index']==index and (v['specification']['role']=='N')==native_role]
            assert len({v['sha256'] for v in rows})==1
    write(base/'smoke-audit.json',dict(passed=True,phases=audits,source=source,binaries=binaries,model=pin(model),native=pin(native)))
    print('All 80 worker smokes pass; no timing verdict.',flush=True)

if __name__=='__main__':main()
