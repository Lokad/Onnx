"""Extract the completed M54 archive with validated in-archive hardlinks as bytes."""
import hashlib,json,sys,tarfile
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/parakeet/wide-entry-first-use-models-amd'))
from run import BASE,prepared
from protocol import pin,read,save
prepared();assert not (BASE/'collection-transfer.json').exists()
target=BASE/'collected';assert target.is_dir() and not any(target.iterdir())
archive=pin(BASE/'results.tar.gz')
with tarfile.open(BASE/'results.tar.gz') as tar:
    members=tar.getmembers();by_name={m.name:m for m in members};assert len(by_name)==len(members)
    for member in members:
        name=PurePosixPath(member.name)
        assert not name.is_absolute() and '..' not in name.parts and '\\' not in member.name
        assert member.isfile() or member.islnk()
        if member.islnk():assert member.linkname in by_name and by_name[member.linkname].isfile()
    receipt=json.load(tar.extractfile(by_name['collection.json']))
    assert receipt['terminal'] and receipt['code']==0 and receipt['input_error'] is None
    assert set(by_name)==set(receipt['files'])|{'collection.json'}
    assert receipt['payload']==pin(BASE/'payload.json')
    for member in members:
        data=tar.extractfile(member).read()
        if member.name!='collection.json':
            assert dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())==receipt['files'][member.name]
        path=(target/member.name).resolve();assert path.is_relative_to(target.resolve()) and not path.exists()
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
for name,wanted in receipt['files'].items():assert pin(target/name)==wanted,name
assert archive==pin(BASE/'results.tar.gz')
save(BASE/'collection-transfer.json',dict(passed=True,archive=archive,receipt=pin(target/'collection.json')))
save(BASE/'collection-hardlink-recovery.json',dict(passed=True,archive=archive,files=len(receipt['files']),hardlinks=sum(m.islnk() for m in members),
    reason='The original regular-file-only extractor refused tar hardlink entries after a complete terminal transfer. Its empty target and complete archive were preserved. Validated same-archive regular-file targets were materialized as byte copies and every file matched the original receipt. No inference, transfer or worker was repeated.'))
print(json.dumps(dict(passed=True,files=len(receipt['files']),hardlinks=sum(m.islnk() for m in members))))
