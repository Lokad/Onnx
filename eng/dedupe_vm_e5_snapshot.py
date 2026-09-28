"""Keep the old E5 model snapshot as a byte-identical hardlink to the canonical model."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/vm-e5-snapshot-dedup-20260928'
loader = importlib.util.spec_from_file_location('snapshot_transport', ROOT/'eng/offload_vm_snapshots.py')
engine = importlib.util.module_from_spec(loader); loader.loader.exec_module(engine)


def script():
    model = ROOT/'models/multilingual-e5-small/model.onnx'
    with model.open('rb') as f:
        expected = dict(bytes=model.stat().st_size, sha256=hashlib.file_digest(f, 'sha256').hexdigest())
    prelude = engine.SCRIPT.split('store=Path(', 1)[0]
    return prelude + '\nEXPECTED=' + repr(expected) + r'''
root=Path('/home/vermorel/Onnx')
source=root/'models/multilingual-e5-small/model.onnx'
target=root/'artifacts/m1-20260918/models/multilingual-e5-small/model.onnx'
def pin(p):
 with p.open('rb') as f:return dict(bytes=p.stat().st_size,sha256=hashlib.file_digest(f,'sha256').hexdigest())
for p in [source,target]:
 assert p.resolve()==p and p.is_relative_to(root) and not p.is_symlink() and pin(p)==EXPECTED
assert source.stat().st_dev==target.stat().st_dev
st=target.stat();assert st.st_nlink==1 and st.st_ino!=source.stat().st_ino
if ACTION=='plan':
 print(json.dumps(dict(passed=True,source=str(source),target=str(target),identity=EXPECTED,
  allocated=st.st_blocks*512,device=st.st_dev,inode=st.st_ino)))
elif ACTION=='apply':
 assert PLAN['source']==str(source) and PLAN['target']==str(target) and PLAN['identity']==EXPECTED
 assert (st.st_dev,st.st_ino,st.st_blocks*512)==(PLAN['device'],PLAN['inode'],PLAN['allocated'])
 link=target.with_name('model.onnx.dedup-link');assert not link.exists() and not link.is_symlink()
 os.link(source,link);os.replace(link,target)
 assert pin(target)==pin(source)==EXPECTED and target.stat().st_ino==source.stat().st_ino
 print(json.dumps(dict(passed=True,allocated_reclaimed=PLAN['allocated'],identity=EXPECTED,
  disk_free=psutil.disk_usage(root).free,original_paths_and_bytes_preserved=True)))
'''


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['plan', 'apply']
    code = script(); identity = hashlib.sha256(code.encode()).hexdigest()
    if sys.argv[1] == 'plan':
        assert not OUT.exists()
        result = json.loads(engine.run.ssh('ACTION="plan"\n'+code, 180))
        OUT.mkdir(); engine.save(OUT/'plan.json', dict(script_sha256=identity, **result))
        print(json.dumps(result))
    else:
        assert not (OUT/'started.json').exists()
        plan = json.loads((OUT/'plan.json').read_text()); assert plan['script_sha256'] == identity
        engine.save(OUT/'started.json', dict(script_sha256=identity))
        result = json.loads(engine.run.ssh('ACTION="apply"\nPLAN='+repr(plan)+'\n'+code, 300))
        engine.save(OUT/'completed.json', result); print(json.dumps(result))
