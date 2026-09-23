"""Four first-use compilation flags over the exact retained M41 source."""
import difflib,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'artifacts/parakeet-first-use-kernels-source-20260923'
PARENT=ROOT/'artifacts/parakeet-short-dispatch-source-v3-20260923'
FILE='src/Lokad.Onnx/MathOps.cs'
METHODS=['PackPanelsB','mm_unsafe_vectorized_intrinsics_2x4packed_bump','mm_unsafe_vectorized_intrinsics_3x4packed','mm_unsafe_vectorized_intrinsics']
ATTRIBUTE='    [MethodImpl(MethodImplOptions.AggressiveOptimization)]\n'
def read(p):return json.loads(p.read_text())
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
def main():
 assert not BASE.exists()
 assert pin(PARENT/'prepared.json')['sha256']=='918a28ab1748887412a432f5d8a3f76289a23a56e6f1761dafb6b95f6aab3fc0'
 parent=read(PARENT/'prepared.json');assert len(parent['source'])==421 and len(parent['before'])==420
 for name,wanted in parent['source'].items():assert pin(PARENT/'source'/name)==wanted,name
 for name,wanted in parent['before'].items():assert pin(ROOT/name)==wanted,name
 for name,digest in [('parakeet-dispatch-events-amd-20260923','c6e1e4d42d3f377f77382a4091ad1150c1a62335a72c7d63a38c728d7a753e19'),('parakeet-dispatch-full-export-amd-20260923','35b18e874e1e0c47a7b9e1fe6f2608b0940c1eb431c289c32ba05855cb2b5afa')]:
  p=ROOT/'artifacts'/name;assert pin(p/'closed.json')['sha256']==digest
  for relative,wanted in read(p/'closed.json')['files'].items():assert pin(p/relative)==wanted,relative
 before=(PARENT/'source'/FILE).read_text();after=before
 for name in METHODS:
  needle='    public unsafe static void '+name+'('
  count=after.count(needle);assert count==(2 if name=='mm_unsafe_vectorized_intrinsics' else 1)
  offset=after.index(needle);header=after[offset:after.index('{',offset)]
  assert 'float*' in header and 'double*' not in header
  after=after[:offset]+ATTRIBUTE+after[offset:]
 assert after.replace(ATTRIBUTE,'')==before.replace(ATTRIBUTE,'')
 assert after.count(ATTRIBUTE)==before.count(ATTRIBUTE)+4
 BASE.mkdir();source=BASE/'source';source.mkdir()
 for name in parent['source']:
  target=source/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(PARENT/'source'/name,target)
 (source/FILE).write_text(after,encoding='utf8',newline='\n')
 changed=[name for name,wanted in parent['source'].items() if pin(source/name)!=wanted];assert changed==[FILE]
 (BASE/'candidate.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=FILE,tofile=FILE)),encoding='utf8')
 shutil.copy2(ROOT/'.agent/m43-parakeet-first-use-kernels-20260923.md',BASE/'prospective-plan.md')
 shutil.copy2(PARENT/'census.json',BASE/'census.json');assert pin(BASE/'census.json')==parent['census']
 result=dict(passed=True,built=False,numerically_qualified=False,root_product_changed=False,parent=pin(PARENT/'prepared.json'),parent_changed=changed,methods=METHODS,implementation_flag=512,before=parent['before'],source={name:pin(source/name) for name in parent['source']},patch=pin(BASE/'candidate.patch'),census=pin(BASE/'census.json'),plan=pin(BASE/'prospective-plan.md'),generator=pin(Path(__file__)),scope='Only four float MethodImpl flags change relative to M41; method bodies and all root product inputs remain exact.')
 for name,wanted in parent['before'].items():assert pin(ROOT/name)==wanted,name
 (BASE/'prepared.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(dict(prepared=pin(BASE/'prepared.json'),patch=result['patch'],parent_changed=changed,methods=METHODS)))
if __name__=='__main__':main()
