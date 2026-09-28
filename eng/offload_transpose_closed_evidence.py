"""Restore native profiling headroom from verified terminal snapshots, within disk limits."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT/'eng/offload_vm_snapshots.py'
assert hashlib.sha256(ENGINE.read_bytes()).hexdigest() == 'e5f9104a14bcf6a2a1b2ccf08c7cb8c600eda966bb0a94f1e0e3ca721960ec98'
loader = importlib.util.spec_from_file_location('transpose_snapshot_offload', ENGINE)
engine = importlib.util.module_from_spec(loader); loader.loader.exec_module(engine)
engine.OUT = ROOT/'artifacts/vm-transpose-closed-evidence-offload-20260928'


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size, sha256=hashlib.file_digest(stream, 'sha256').hexdigest())


def configure():
    roots = {}
    campaigns = [(f'parakeet-transpose-axis-{lane}-amd-20260928', f'lokad-transpose-axis-{lane}-20260928')
                 for lane in ['models', 'app', 'shared', 'pyannote', 'graphs', 'pyannote-app', 'root']]
    campaigns += [(f'parakeet-attention-owned-{lane}-amd-20260928', f'lokad-attention-owned-{lane}-20260928')
                  for lane in ['app', 'pyannote-app']]
    campaigns += [(f'parakeet-{family}-root-amd-20260927', f'lokad-parakeet-{family}-root-20260927')
                  for family in ['decoder-packed-row', 'rational-sigmoid']]
    for local, remote in campaigns:
        folder = ROOT/'artifacts'/local
        proof_path = folder/'closed.json'; proof = json.loads(proof_path.read_text())
        assert proof['passed']
        receipt_path = folder/'collected/collection.json'; receipt_pin = pin(receipt_path)
        assert receipt_pin == proof.get('collection', proof.get('files', {}).get('collected/collection.json'))
        receipt = json.loads(receipt_path.read_text())
        assert receipt['terminal'] and receipt['code'] == 0 and receipt.get('input_error') is None
        for name, wanted in receipt['files'].items():
            assert pin(folder/'collected'/name) == wanted, name
        roots['/dev/shm/'+remote] = dict(collections={'collection.json':dict(pin=receipt_pin,code=0)},
                                        local_proof=pin(proof_path))
    # Both focused builds are terminal. Preserve the failed first build's code,
    # and bind both build and contract owners for the successful recovery.
    for suffix, failed in [('', True), ('-recovery', False)]:
        folder = ROOT/f'artifacts/parakeet-transpose-axis-build{suffix}-amd-20260928'
        proof_path = folder/('failed.json' if failed else 'closed.json')
        proof = json.loads(proof_path.read_text())
        assert proof['passed'] == (not failed)
        collections = {}
        for kind in (['build'] if failed else ['build', 'capture']):
            collected = folder/(kind+'-collected'); receipt_path = collected/(kind+'-collection.json')
            receipt_pin = pin(receipt_path); receipt = json.loads(receipt_path.read_text())
            code = 1 if failed else 0
            assert receipt['terminal'] and receipt['code'] == code and receipt.get('input_error') is None
            for name, wanted in receipt['files'].items():assert pin(collected/name) == wanted, name
            collections[kind+'-collection.json'] = dict(pin=receipt_pin,code=code)
        assert proof['collection'] == receipt_pin
        if not failed:
            assert pin(folder/'capture-collected/build-collection.json') == collections['build-collection.json']['pin']
        roots[f'/dev/shm/lokad-transpose-axis-build{suffix}-20260928'] = dict(
            collections=collections,local_proof=pin(proof_path))
    guard = '\nROOTS=' + repr(roots) + r'''
for name, wanted in ROOTS.items():
 folder=Path(name)
 assert folder.parent==Path('/dev/shm') and folder.resolve()==folder
 for relative,expected in wanted['collections'].items():
  path=folder/relative;assert pin(path)==expected['pin']
  receipt=json.loads(path.read_text())
  assert receipt['terminal'] and receipt['code']==expected['code'] and receipt.get('input_error') is None
  for identity in receipt['identities']:
   try:
    process=psutil.Process(identity['pid'])
    assert process.create_time()!=identity['birth'] or process.status()==psutil.STATUS_ZOMBIE
   except psutil.NoSuchProcess:pass
'''
    original = engine.SCRIPT
    start = original.index('def eligible(p):'); end = original.index('\ndef resources():', start)
    eligible = '''def eligible(p):
 if p.is_symlink() or not p.is_file() or p.resolve()!=p:return False
 parts=p.relative_to('/dev/shm').parts
 if len(parts)<2 or str(Path('/dev/shm')/parts[0]) not in ROOTS:return False
 if p.stat().st_nlink!=1 or p.stat().st_size<4096:return False
 return parts[1] not in {'runtime','runtimes','built','measured','models','assets','tools','nuget-feed'}
'''
    script = original[:start] + guard + eligible + original[end:]
    script = script.replace('tmpfs-snapshots-20260927', 'transpose-closed-evidence-20260928')
    before = ' assert rows\n result=dict(files=rows'
    assert script.count(before) == 1
    after = ''' assert rows
 # Retain the original 12 GiB profiling and 2 GiB disk limits, with margins.
 current=resources()
 needed=max(0,12*1024**3+32*1024**2-current['available'])
 capacity=current['disk']-2*1024**3-16*1024**2
 assert needed>0 and capacity>=needed,(needed,capacity,current)
 selected=[];allocated=0
 for row in sorted(rows,key=lambda r:(-r['allocated'],r['path'])):
  if allocated+row['allocated']>capacity:continue
  selected.append(row);allocated+=row['allocated']
  if allocated>=needed:break
 assert allocated>=needed,(allocated,needed)
 rows=selected
 result=dict(files=rows'''
    script = script.replace(before, after)
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
