"""Deploy once, verify inputs, freeze and launch the existing bounded supervisor."""
from pathlib import Path
import json, subprocess
from prepare import BASE,pin,read,write

KEY='C:/Users/JoannesVermorel/.ssh/id_onnx-bench.pem'
HOST='vermorel@74.178.91.76'
REMOTE='/home/vermorel/Onnx/artifacts/pyannote-frame-meetings-20260920'


def ssh(script):
    return subprocess.check_output(['ssh','-i',KEY,'-o','BatchMode=yes','-o','ConnectTimeout=15',HOST,'python3 -B -'],input=script,text=True,encoding='utf-8')


def main():
    transfer=read(BASE/'transfer.json');assert pin(BASE/'payload.tar.gz')==transfer['archive']
    assert not (BASE/'deployment.json').exists()
    ssh("from pathlib import Path\nimport shutil\nbase=Path(%r)\nassert shutil.disk_usage(str(base.parent)).free>64*1024**2\nbase.mkdir()"%REMOTE)
    subprocess.run(['scp','-i',KEY,'-o','BatchMode=yes',str(BASE/'payload.tar.gz'),HOST+':'+REMOTE+'/payload.tar.gz'],check=True)
    script='''from pathlib import Path
import hashlib,json,tarfile,subprocess,sys,os
base=Path(%r);root=base.parents[1];old=root/'artifacts/pyannote-natural-meetings-20260920'
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
assert pin(base/'payload.tar.gz')==%r
with tarfile.open(base/'payload.tar.gz') as tar:tar.extractall(base,filter='data')
prepared=json.loads((base/'preparation.json').read_text())
for name in ['ES2004a-600s.wav','IS1009a-600s.wav']:os.link(old/'inputs'/name,base/'inputs'/name)
for name,wanted in prepared['files'].items():assert pin(base/name)==wanted,name
manifest=json.loads((base/'manifest.json').read_text())
for item in manifest['models'].values():assert pin(root/item['amd_path'])=={k:item[k] for k in ['bytes','sha256']}
env={k:v for k,v in os.environ.items() if not k.lower().startswith(('lokad_','dotnet_','complus_'))}
env['PYTHONPATH']=str(root/'artifacts/asr-multilingual-amd-20260920/python')
command=['python3','-B',str(base/'runtime/supervise.py')]
input_result=subprocess.run(command+['run','--artifact',str(base),'--engine','managed','--mode','inputs'],env=env,cwd=root,timeout=180,capture_output=True,text=True)
with (base/'input-launch.json').open('x') as f:json.dump(dict(code=input_result.returncode,stdout=input_result.stdout,stderr=input_result.stderr),f,indent=2)
assert input_result.returncode==0,input_result.stderr
inputs=json.loads((base/'process-managed-inputs/worker/inputs.json').read_text());assert inputs['passed'] and inputs['affinity']==4
assert inputs['cases']==[{k:c[k] for k in ['name','samples','pcm_sha256']} for c in manifest['cases']]
files=dict(prepared['files'])
for name in ['preparation.json','input-launch.json','process-managed-inputs/worker/inputs.json','process-managed-inputs/identity.json','process-managed-inputs/complete.json']:
 files[name]=pin(base/name)
frozen=dict(schema=1,source_commit=prepared['source'],product_source=prepared['product_source'],files=files,native_files={},limits=manifest['limits'],
 schedule=dict(managed=[c['name'] for c in manifest['cases']],native='retained original outputs'))
with (base/'frozen.json').open('x') as f:json.dump(frozen,f,indent=2)
output=subprocess.check_output(command+['launch','--artifact',str(base),'--engine','managed'],env=env,cwd=root,text=True)
print(json.dumps(dict(deployment=json.loads(output),frozen=pin(base/'frozen.json'),input_check=inputs)))
'''%(REMOTE,transfer['archive'])
    result=json.loads(ssh(script));write(BASE/'deployment.json',result);print(json.dumps(result))


if __name__=='__main__':main()
