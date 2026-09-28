"""Move verified terminal campaign evidence from tmpfs to disk without deleting it."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'eng/offload_vm_snapshots.py'
loader = importlib.util.spec_from_file_location('closed_evidence_offload', ENGINE)
engine = importlib.util.module_from_spec(loader); loader.loader.exec_module(engine)
engine.OUT = ROOT/'artifacts/vm-attention-closed-evidence-offload-20260928'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def configure():
    roots = {}
    campaigns = [('attention-owned', 'attention-owned', '20260928', lane)
                 for lane in ['root', 'root-recovery', 'models', 'shared', 'pyannote', 'graphs']]
    campaigns += [(family, prefix, '20260927', lane)
                  for family, prefix, lanes in [
                      ('decoder-lstm-layout', 'lstmlayout', ['root', 'ort-profile']),
                      ('pointwise-tail', 'pwt', ['root', 'ort-profile', 'models', 'shared', 'pyannote'])]
                  for lane in lanes]
    for family, prefix, date, lane in campaigns:
        folder = ROOT/f'artifacts/parakeet-{family}-{lane}-amd-{date}'
        failed = family == 'attention-owned' and lane == 'root'
        proof_path = folder/('failed.json' if failed else 'closed.json')
        proof = json.loads(proof_path.read_text())
        assert proof['passed'] == (not failed)
        receipt_path = folder/'collected/collection.json'
        receipt_pin = pin(receipt_path)
        assert receipt_pin == proof.get('collection', proof.get('files', {}).get('collected/collection.json'))
        receipt = json.loads(receipt_path.read_text())
        assert receipt['terminal'] and receipt['code'] == (1 if failed else 0)
        assert receipt.get('input_error') is None
        for name, wanted in receipt['files'].items(): assert pin(folder/'collected'/name) == wanted, name
        roots[f'/dev/shm/lokad-{prefix}-{lane}-{date}'] = dict(
            collection=receipt_pin, code=receipt['code'], local_proof=pin(proof_path))
    guard = '\nROOTS=' + repr(roots) + r'''
for name, wanted in ROOTS.items():
 folder=Path(name)
 assert folder.parent==Path('/dev/shm') and folder.resolve()==folder
 path=folder/'collection.json';assert pin(path)==wanted['collection']
 receipt=json.loads(path.read_text())
 assert receipt['terminal'] and receipt['code']==wanted['code'] and receipt.get('input_error') is None
 for identity in receipt['identities']:
  try:
   process=psutil.Process(identity['pid'])
   assert process.create_time()!=identity['birth'] or process.status()==psutil.STATUS_ZOMBIE
  except psutil.NoSuchProcess:pass
'''
    original = engine.SCRIPT
    start = original.index('def eligible(p):')
    end = original.index('\ndef resources():', start)
    eligible = '''def eligible(p):
 if p.is_symlink() or not p.is_file() or p.resolve()!=p:return False
 parts=p.relative_to('/dev/shm').parts
 if len(parts)<2 or str(Path('/dev/shm')/parts[0]) not in ROOTS:return False
 if p.stat().st_nlink!=1 or p.stat().st_size<4096:return False
 # Retain live-use input/runtime locations in RAM. Move only terminal campaign
 # snapshots, package caches and recorded outputs; every original path remains.
 return parts[1] not in {'runtime','runtimes','built','measured','models','assets','tools','nuget-feed'}
'''
    script = original[:start] + guard + eligible + original[end:]
    script = script.replace('tmpfs-snapshots-20260927', 'attention-closed-evidence-20260928')
    engine.SCRIPT = script
    return dict(adapter=pin(Path(__file__)), engine=pin(ENGINE), campaigns=roots,
                remote_script_sha256=hashlib.sha256(script.encode()).hexdigest())


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1] in ['plan', 'apply']
    configuration = configure()
    if sys.argv[1] == 'plan':
        engine.main('plan')
        engine.save(engine.OUT/'configuration.json', configuration)
    else:
        assert json.loads((engine.OUT/'configuration.json').read_text()) == configuration
        engine.main('apply')
