"""Retire locally retained, closed output copies before the unstarted M78 profiles."""
import json
from pathlib import Path
import retire_closed_output_duplicates as common

ROOT=common.ROOT
OUT=ROOT/'artifacts/parakeet-profile-headroom-retention-20260925'
SOURCES=['e5-frequency-diagnostic','parakeet-packed-final-row-shared',
         'parakeet-packed-final-row-pyannote','parakeet-packed-final-row-graphs']


def main():
    assert not OUT.exists()
    rows=[]
    for name in SOURCES:
        local=ROOT/('artifacts/'+name+'-amd-20260925')
        proof=common.read(local/'closed.json');assert proof['passed']
        for relative,wanted in proof['files'].items():assert common.pin(local/relative)==wanted,relative
        collection=common.read(local/'collected/collection.json')
        assert collection['terminal'] and collection['code']==0 and collection['input_error'] is None
        for relative,wanted in collection['files'].items():
            eligible=(relative in ['a-capture/capture.nettrace','b-capture/capture.nettrace',
                'a-export/events/events.jsonl.gz','b-export/events/events.jsonl.gz']
                if name=='e5-frequency-diagnostic' else relative.endswith('.f32') and '/output/' in relative)
            if not eligible:continue
            path=local/'collected'/relative
            assert common.pin(path)==wanted==proof['files']['collected/'+relative]
            rows.append(dict(root='/dev/shm/lokad-'+name+'-20260925',name=relative,
                local=path.relative_to(ROOT).as_posix(),identity=wanted,
                collection=common.pin(local/'collected/collection.json'),owners=collection['identities'],
                closure=common.pin(local/'closed.json')))
    script=common.COMMON+f'''
rows=json.loads(zlib.decompress(base64.b64decode({common.encoded(rows)})))
selected={{}};skipped=[];checked=set()
for row in rows:
 root=Path(row['root']);path=root/row['name']
 assert root.parent==Path('/dev/shm') and root.resolve()==root
 assert path.resolve()==path and path.is_relative_to(root) and not path.is_symlink()
 if not path.exists() or str(path) in protected or path.stat().st_nlink!=1:
  skipped.append(str(path));continue
 if str(root) not in checked:
  assert pin(root/'collection.json')==row['collection']
  assert not any(live(i) for i in row['owners']);checked.add(str(root))
 assert pin(path)==row['identity']
 selected[str(path)]=dict(**row,physical=path.stat().st_blocks*512)
physical=sum(r['physical'] for r in selected.values())
assert selected and physical<=256*1024**2
idle()
'''
    snapshot=json.loads(common.ssh(script+'''
print(json.dumps(dict(passed=True,selected=selected,skipped=skipped,physical=physical,manifests=manifests)))
''',300))
    OUT.mkdir();common.save(OUT/'prospective.json',dict(**snapshot,tool=common.pin(__file__),common=common.pin(common.__file__)))
    result=json.loads(common.ssh(script+f'''
assert selected==json.loads(zlib.decompress(base64.b64decode({common.encoded(snapshot['selected'])})))
assert manifests=={snapshot['manifests']!r} and physical=={snapshot['physical']}
before=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free)
for path in selected:Path(path).unlink()
assert all(not Path(path).exists() for path in selected)
idle()
print(json.dumps(dict(passed=True,files=len(selected),physical_bytes=physical,
 logical_bytes=sum(r['identity']['bytes'] for r in selected.values()),before=before,
 after=dict(available=psutil.virtual_memory().available,tmpfs=psutil.disk_usage('/dev/shm').free))))
''',300))
    for row in snapshot['selected'].values():assert common.pin(ROOT/row['local'])==row['identity']
    common.save(OUT/'closed.json',dict(**result,prospective=common.pin(OUT/'prospective.json'),all_local_originals_retained=True))
    print(json.dumps(result))


if __name__=='__main__':main()
