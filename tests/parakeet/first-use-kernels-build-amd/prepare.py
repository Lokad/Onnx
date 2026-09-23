"""Freeze M43 source and explicit current/M41 metadata comparisons."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-first-use-kernels-build-amd-20260923'
SOURCE=ROOT/'artifacts/parakeet-first-use-kernels-source-20260923'
CURRENT=ROOT/'artifacts/parakeet-winograd-baseline-amd-20260923'
PARENT=ROOT/'artifacts/parakeet-short-dispatch-build-amd-v3-20260923'
LATEST=ROOT/'artifacts/parakeet-dispatch-full-export-amd-20260923'
QUALIFIED=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
def previous_closed():
 for folder,digest in [(PARENT,'0a2d96454588d8e90d246ccbf91c1069d0e1efdd88710b16985dda89f42b3d39'),(CURRENT,'2e75c249ca3f76fc90c0179e2244cd677e829cf14da18029ec73f0a2ed03abf3'),(LATEST,'35b18e874e1e0c47a7b9e1fe6f2608b0940c1eb431c289c32ba05855cb2b5afa'),(QUALIFIED,'62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0')]:
  assert pin(folder/'closed.json')['sha256']==digest
  value=read(folder/'closed.json');assert value['passed']
  for name,wanted in value['files'].items():assert pin(folder/name)==wanted,name
 assert pin(SOURCE/'prepared.json')['sha256']=='829e26d7acd55ccf969f4292949abc19385a014f562517344a3265d42a0f51c0'
 source=read(SOURCE/'prepared.json');assert source['passed'] and not source['root_product_changed'] and not source['built']
 assert len(source['source'])==421 and len(source['before'])==420 and source['parent_changed']==['src/Lokad.Onnx/MathOps.cs']
 for name,wanted in source['source'].items():assert pin(SOURCE/'source'/name)==wanted,name
 for name,wanted in source['before'].items():assert pin(ROOT/name)==wanted,name
 for name,key in [('candidate.patch','patch'),('prospective-plan.md','plan'),('census.json','census')]:assert pin(SOURCE/name)==source[key]
 assert pin(ROOT/'tests/parakeet/first-use-kernels-source/prepare.py')==source['generator']
def prepare():
 assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
 def copy(p,q):
  q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);originals[p.relative_to(ROOT).as_posix()]=pin(p)
 source=read(SOURCE/'prepared.json')
 for name in source['source']:copy(SOURCE/'source'/name,bundle/'source'/name)
 for name in ['protocol.py','remote.py','remote_prepare.py','checks.py','base_checks.py']:copy(TOOLS/name,bundle/'tools'/name)
 copy(TOOLS/'Bridge.cs.txt',bundle/'bridge-source/Program.cs');copy(TOOLS/'Bridge.csproj',bundle/'bridge-source/Bridge.csproj')
 for folder,label in [(CURRENT,'current'),(QUALIFIED,'qualified'),(PARENT,'previous')]:
  for name in ['closed.json','analysis.json','payload.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
  copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
 for name in ['closed.json']:copy(LATEST/name,bundle/'evidence'/('latest-'+name))
 copy(LATEST/'collected/collection.json',bundle/'evidence/latest-collection.json')
 copy(SOURCE/'prepared.json',bundle/'evidence/source-prepared.json');copy(SOURCE/'candidate.patch',bundle/'candidate.patch');copy(SOURCE/'census.json',bundle/'evidence/census.json')
 shutil.copy2(ROOT/'.agent/m43-parakeet-first-use-kernels-20260923.md',bundle/'prospective-plan.md')
 current=CURRENT/'collected/runtimes/current';previous=PARENT/'collected/runtime'
 stage=dict(passed=True,source_prepared=pin(SOURCE/'prepared.json'),measured={n:pin(current/n) for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},previous={n:pin(previous/n) for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']},measured_files={p.name:pin(p) for p in current.iterdir() if p.is_file()},previous_files={p.name:pin(p) for p in previous.iterdir() if p.is_file()},files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()})
 assert sum(n.startswith('source/') for n in stage['files'])==421;save(bundle/'stage.json',stage)
 for p in TOOLS.iterdir():
  if p.is_file():
   if p.suffix=='.py':ast.parse(p.read_text(),str(p))
   originals[p.relative_to(ROOT).as_posix()]=pin(p)
 with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
  for p in sorted(bundle.rglob('*')):
   if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
 save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
 print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),source=stage['source_prepared'])))
if __name__=='__main__':prepare()
