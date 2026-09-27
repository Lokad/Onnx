"""Restore profiling headroom by relocating six closed campaigns' snapshots."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'eng/offload_vm_snapshots.py'
loader = importlib.util.spec_from_file_location('snapshot_offload', ENGINE)
engine = importlib.util.module_from_spec(loader)
loader.loader.exec_module(engine)
engine.OUT = ROOT / 'artifacts/vm-pointwise-closed-snapshot-offload-20260927'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def configure():
    receipts = {}
    for family, prefix in [('decoder-lstm-layout', 'lstmlayout'), ('pointwise-tail', 'pwt')]:
        for lane in ['root', 'graphs', 'pyannote-app']:
            folder = ROOT / f'artifacts/parakeet-{family}-{lane}-amd-20260927'
            closure = json.loads((folder / 'closed.json').read_text())
            receipt = folder / 'collected/collection.json'
            assert closure['passed'] and closure['files']['collected/collection.json'] == pin(receipt)
            value = json.loads(receipt.read_text())
            assert value['terminal'] and value['code'] == 0 and value['input_error'] is None
            receipts[f'/dev/shm/lokad-{prefix}-{lane}-20260927'] = pin(receipt)
    guard = '\nRECEIPTS=' + repr(receipts) + r'''
for root, wanted in RECEIPTS.items():
 folder=Path(root)
 assert folder.parent==Path('/dev/shm') and folder.resolve()==folder
 path=folder/'collection.json';assert pin(path)==wanted
 receipt=json.loads(path.read_text())
 assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
 for identity in receipt['identities']:
  try:
   process=psutil.Process(identity['pid'])
   assert process.create_time()!=identity['birth'] or process.status()==psutil.STATUS_ZOMBIE
  except psutil.NoSuchProcess:pass
'''
    script = engine.SCRIPT.replace('tmpfs-snapshots-20260927', 'pointwise-closed-snapshots-20260927')
    script = script.replace('\ndef eligible(p):', guard + '\ndef eligible(p):')
    old = "if len(parts)<2 or not parts[0].startswith('lokad-') or parts[0].startswith('lokad-lstmlayout'):return False"
    assert old in script
    script = script.replace(old, "if len(parts)<2 or str(Path('/dev/shm')/parts[0]) not in RECEIPTS:return False")
    engine.SCRIPT = script
    return dict(adapter=pin(Path(__file__)), engine=pin(ENGINE), receipts=receipts,
                remote_script_sha256=hashlib.sha256(script.encode()).hexdigest())


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['plan', 'apply']
    configuration = configure()
    if sys.argv[1] == 'plan':
        engine.main('plan')
        engine.save(engine.OUT / 'configuration.json', configuration)
    else:
        assert json.loads((engine.OUT / 'configuration.json').read_text()) == configuration
        engine.main('apply')
