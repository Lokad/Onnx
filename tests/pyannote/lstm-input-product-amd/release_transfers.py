"""Reclaim two redundant upload archives; preserve all extracted inputs and evidence."""
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/pyannote/lstm-input-screen-amd'))
import run
from protocol import pin,read,save


def main():
    base=ROOT/'artifacts/pyannote-lstm-product-space-20260922';assert not base.exists()
    rows=[]
    for local,remote,closure in [
        ('pyannote-blocked-spatial-product-amd-20260922','lokad-pyannote-blocked-spatial-product-20260922','failure-closed.json'),
        ('pyannote-vector-input-layout-amd-20260922','lokad-pyannote-vector-input-layout-20260922','closed.json')]:
        folder=ROOT/'artifacts'/local;proof=read(folder/closure)
        for name,wanted in proof['files'].items():assert pin(folder/name)==wanted,name
        archive=pin(folder/'payload.tar.gz');assert archive==read(folder/'prepared.json')['archive']
        rows.append(dict(remote='/dev/shm/'+remote,local_archive=str(folder/'payload.tar.gz'),archive=archive,
            closure=pin(folder/closure),collection=pin(folder/'collected/collection.json')))
    base.mkdir();save(base/'prospective.json',dict(files=rows,tool=pin(__file__)))
    script=run.PRELUDE+f'''
from protocol import pin,read,save
from remote import idle,live
idle();rows={rows!r}
receipt=Path('/dev/shm/lokad-pyannote-lstm-product-space-20260922.json')
assert not receipt.exists()
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for row in rows:
 folder=Path(row['remote']);archive=folder/'transfer.tar.gz'
 assert folder.resolve()==folder and archive.resolve()==archive and archive.is_file() and not archive.is_symlink()
 assert pin(archive)==row['archive'] and pin(folder/'collection.json')==row['collection']
 collection=read(folder/'collection.json')
 assert collection['terminal'] and 'transfer.tar.gz' not in collection['files']
 assert not any(live(i) for i in collection['identities'])
 for root in Path('/dev/shm').glob('lokad-*'):
  if not root.is_dir():continue
  for name in ['payload.json','execution/execution.json']:
   p=root/name
   if p.exists():assert str(archive) not in p.read_text(),str(p)
for row in rows:
 archive=Path(row['remote'])/'transfer.tar.gz'
 assert pin(archive)==row['archive']
 archive.unlink()
result=dict(passed=True,files=rows,bytes_reclaimed=sum(r['archive']['bytes'] for r in rows),before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free),
 extracted_inputs_untouched=True,local_archives_preserved=True)
save(receipt,result);print(json.dumps(result))
'''
    (base/'remote-script.py').write_text(script,encoding='utf8')
    value=json.loads(run.ssh(script));save(base/'receipt.json',value)
    for row in rows:assert pin(row['local_archive'])==row['archive']
    print(json.dumps(value))


if __name__=='__main__':main()
