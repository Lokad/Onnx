"""Reuse the qualified full-model transport with only new bindings and namespace."""
import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];TOOLS=Path(__file__).resolve().parent
ORIGINAL=TOOLS.parent/'observed-dense-where-models-amd'
sys.path.append(str(ORIGINAL))
loader=importlib.util.spec_from_file_location('retained_model_transport',ORIGINAL/'run.py')
retained=importlib.util.module_from_spec(loader);loader.loader.exec_module(retained)
BASE=ROOT/'artifacts/parakeet-direct-depthwise-models-amd-20260925'
REMOTE='/dev/shm/lokad-parakeet-direct-depthwise-models-20260925'
retained.PRELUDE=retained.PRELUDE.replace(retained.REMOTE,REMOTE)
retained.BASE,retained.REMOTE=BASE,REMOTE
prepared,prepare,stage,launch,observe,collect=retained.prepared,retained.prepare,retained.stage,retained.launch,retained.observe,retained.collect
ssh,PRELUDE,pin,read,save=retained.ssh,retained.PRELUDE,retained.pin,retained.read,retained.save

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['prepare','stage','launch','observe','collect']
    globals()[sys.argv[1]]()
