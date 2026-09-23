"""Freeze a diagnostic-only comparison; both rejected scores remain unchanged."""
import ast,json,shutil,tarfile
from pathlib import Path
from protocol import pin,read,save
ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
BASE=ROOT/'artifacts/parakeet-dispatch-events-amd-20260923'
SCREEN=ROOT/'artifacts/parakeet-short-dispatch-screen-amd-20260923'
BUILD=ROOT/'artifacts/parakeet-short-dispatch-build-amd-v3-20260923'
TRACER=ROOT/'artifacts/parakeet-current-profile-amd-20260923/payload/tracer'
ROOT_PROOF=ROOT/'artifacts/pyannote-winograd-product-root-amd-20260923'
def previous_closed():
 for folder,digest in [(SCREEN,'59da0c26aaad78f43caf1401e0fc70148db8d3f9634921340f6c3b27ce17f2d4'),(BUILD,'0a2d96454588d8e90d246ccbf91c1069d0e1efdd88710b16985dda89f42b3d39')]:
  assert pin(folder/'closed.json')['sha256']==digest
  value=read(folder/'closed.json');assert value['passed']
  for name,wanted in value['files'].items():assert pin(folder/name)==wanted,name
 assert not read(SCREEN/'analysis.json')['admitted']
 assert pin(ROOT_PROOF/'closed.json')['sha256']=='62141a2a722548697c106e42b2c0d9425b4f0c6ce166611a5bc3ca26a4fccdd0'
 before={n.removeprefix('source/'):v for n,v in read(ROOT_PROOF/'bundle/stage.json')['files'].items() if n.startswith('source/')};assert len(before)==420
 for name,wanted in before.items():assert pin(ROOT/name)==wanted,name
 # Tracer files were already part of the qualified profile payload.
 manifest=read(TRACER.parent/'payload.json')
 for p in TRACER.rglob('*'):
  if p.is_file():assert pin(p)==manifest['files'][p.relative_to(TRACER.parent).as_posix()]
def prepare():
 assert not BASE.exists();previous_closed();BASE.mkdir();bundle=BASE/'bundle';bundle.mkdir();originals={}
 def copy(p,q):
  q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);originals[p.relative_to(ROOT).as_posix()]=pin(p)
 for source,target in [('Driver.cs','consumer/Driver.cs'),('Producer.csproj','consumer/Producer.csproj'),('Export.cs','exporter/Export.cs'),('Exporter.csproj','exporter/Exporter.csproj')]:copy(TOOLS/source,bundle/'source'/target)
 copy(ROOT/'global.json',bundle/'source/global.json')
 for name in ['protocol.py','remote.py','remote_prepare.py']:copy(TOOLS/name,bundle/'tools'/name)
 copy(TOOLS/'README.md',bundle/'README.md');shutil.copy2(ROOT/'.agent/m42-parakeet-dispatch-events-20260923.md',bundle/'prospective-plan.md')
 for folder in ['runtimes/current','runtimes/candidate','fixtures']:
  for p in (SCREEN/'bundle'/folder).rglob('*'):
   if p.is_file():
    relative=p.relative_to(SCREEN/'bundle').as_posix();assert pin(p)==read(SCREEN/'payload.json')['files'][relative]
    copy(p,bundle/relative)
 for p in TRACER.rglob('*'):
  if p.is_file():copy(p,bundle/'tracer'/p.relative_to(TRACER))
 for label,folder in [('screen',SCREEN),('build',BUILD)]:
  for name in ['closed.json','payload.json']:copy(folder/name,bundle/'evidence'/(label+'-'+name))
  copy(folder/'collected/collection.json',bundle/'evidence'/(label+'-collection.json'))
 products=read(SCREEN/'payload.json')['products']
 save(bundle/'stage.json',dict(passed=True,products=products,diagnostic_only=True,files={p.relative_to(bundle).as_posix():pin(p) for p in bundle.rglob('*') if p.is_file()}))
 for p in TOOLS.iterdir():
  if p.is_file():
   if p.suffix=='.py':ast.parse(p.read_text(),str(p))
   originals[p.relative_to(ROOT).as_posix()]=pin(p)
 with tarfile.open(BASE/'payload.tar.gz','w:gz') as tar:
  for p in sorted(bundle.rglob('*')):
   if p.is_file():tar.add(p,arcname=p.relative_to(bundle).as_posix(),recursive=False)
 save(BASE/'prepared.json',dict(passed=True,files=originals,stage=pin(bundle/'stage.json'),archive=pin(BASE/'payload.tar.gz')))
 print(json.dumps(dict(archive=pin(BASE/'payload.tar.gz'),stage=pin(bundle/'stage.json'),products=products)))
if __name__=='__main__':prepare()
