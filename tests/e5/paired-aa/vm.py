"""Digest-verified A/A deployment and collection; polling never mutates the campaign."""
from pathlib import Path
import argparse,importlib.util

directory=Path(__file__).resolve().parent;support=directory/'vm_support.py'
if not support.exists():support=directory.parents[1]/'whisper/maximum-speech/vm.py'
spec=importlib.util.spec_from_file_location('paired_vm_support',support);protocol=importlib.util.module_from_spec(spec);spec.loader.exec_module(protocol)
protocol.REMOTE='/home/vermorel/Onnx/artifacts/e5-paired-aa-v2-20260920'

def poll(base):
    print(protocol.ssh('''from pathlib import Path
import json,time
b=Path(REMOTE);j=json.loads((b/'result/identity.json').read_text())
print(json.dumps(dict(complete=j['complete'],error=j.get('error'),elapsed=time.time()-j['started'],runs=j['runs']),indent=2))
for item in [j['supervisor']]+j['runs'][-1:]:
    path=Path('/proc')/str(item['pid'])/'stat'
    if path.exists():
        parts=path.read_text().split(') ',1)[1].split();print('PROCESS',item['pid'],'expected',item['start'],'observed',parts[19],'state',parts[0])
    else:print('PROCESS',item['pid'],'absent')
if j['runs']:
    name=j['runs'][-1]['name']
    print('OUTPUT',(b/'result'/(name+'.stdout')).read_text()[-1000:])
    print('ERROR',(b/'result'/(name+'.stderr')).read_text()[-1000:])
'''.replace('REMOTE',repr(protocol.REMOTE))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=('deploy','poll','collect'));p.add_argument('--artifact',type=Path,required=True)
    a=p.parse_args()
    if a.action=='poll':poll(a.artifact.resolve())
    else:getattr(protocol,a.action)(a.artifact.resolve())
