"""Verify the complete extracted e5 collection before retiring its duplicate transport."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/e5-randomized-processes-20260921'
OUT=ROOT/'artifacts/e5-randomized-transfer-retention-20260924'
ARCHIVE=BASE/'results-aa.tar.gz'
EXPECTED=dict(bytes=876104825,sha256='083027c0fbbb57162222aedd9b6cdf223639bca5b01701ce80ab5c803243b42c')


def pin(path):
    with path.open('rb') as stream:
        return dict(bytes=path.stat().st_size,sha256=hashlib.file_digest(stream,'sha256').hexdigest())


def read(path):return json.loads(path.read_text(encoding='utf8'))


def write(path,value):
    with path.open('x',encoding='utf8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def main():
    assert not OUT.exists()
    assert ARCHIVE.resolve()==ARCHIVE and ARCHIVE.is_relative_to(ROOT/'artifacts') and not ARCHIVE.is_symlink()
    assert not subprocess.check_output(['git','ls-files','--',ARCHIVE.relative_to(ROOT).as_posix()],cwd=ROOT)
    assert pin(ARCHIVE)==EXPECTED
    transfer=read(BASE/'collection-transfer-aa.json')
    collection_path=BASE/'collected-aa/aa-collection.json';collection=read(collection_path)
    verification=read(BASE/'aa-verification.json')
    assert transfer['passed'] and transfer['archive']==EXPECTED and transfer['receipt']==pin(collection_path)
    assert collection['terminal'] and collection['code']==0 and collection['remote_audit_code']==0
    assert verification['passed'] and verification['collection']==pin(collection_path)
    assert not verification['statistical_screen'] and not verification['diagnostic_screen']
    assert (verification['measured_calls'],verification['conditioning_calls'])==(334080,2708171)
    assert transfer['files']==len(collection['files'])==19839
    assert transfer['births']==collection['births']
    retained=BASE/'collected-aa'
    expected=dict(collection['files']);expected['aa-collection.json']=pin(collection_path)
    assert {p.relative_to(retained).as_posix() for p in retained.rglob('*') if p.is_file()}==set(expected)
    seen=set();total=0
    with tarfile.open(ARCHIVE,'r|gz') as archive:
        for member in archive:
            assert member.isfile() and member.name not in seen and member.name in expected
            path=Path(member.name)
            assert not path.is_absolute() and '..' not in path.parts
            target=retained/path
            assert target.resolve().is_relative_to(retained.resolve()) and not target.is_symlink()
            wanted=expected[member.name];assert member.size==wanted['bytes']
            with archive.extractfile(member) as stream:
                digest=hashlib.file_digest(stream,'sha256').hexdigest()
            assert digest==wanted['sha256'] and pin(target)==wanted,member.name
            seen.add(member.name);total+=member.size
    assert seen==set(expected) and len(seen)==19840
    # Preserve every raw result and all existing report/verification bytes.
    evidence={name:pin(BASE/name) for name in ['collection-transfer-aa.json','aa-verification.json','aa-analysis.json','aa-report.json']}
    report=ROOT/'tests/e5/randomized-processes/aa-results.md'
    observations=ROOT/'tests/e5/randomized-processes/aa-observations.json'
    assert pin(report)==verification['report'] and pin(observations)==verification['observations']
    assert pin(BASE/'collected-aa/frozen.json')==verification['frozen']==collection['frozen']
    assert pin(ARCHIVE)==EXPECTED
    OUT.mkdir()
    write(OUT/'verification.json',dict(passed=True,archive=EXPECTED,
        collection=pin(collection_path),members=len(seen),retained_bytes=total,
        evidence=evidence,report=pin(report),observations=pin(observations),generator=pin(Path(__file__)),
        terminal=True,original_failed_verdict_preserved=True,all_raw_files_retained=True,
        operation='Remove only the duplicate transport archive; its complete extracted collection remains byte-exact.'))
    metadata=ARCHIVE.stat()
    write(OUT/'manifest.json',dict(repo=str(ROOT),keep=[],extensions=['.gz'],planned_bytes=EXPECTED['bytes'],
        verification=pin(OUT/'verification.json'),files=[dict(path=ARCHIVE.relative_to(ROOT).as_posix(),
        bytes=metadata.st_size,mtime_ns=metadata.st_mtime_ns)]))
    print(json.dumps(dict(passed=True,archive=EXPECTED,verified_members=len(seen),retained_bytes=total,
        manifest=str(OUT/'manifest.json'))))


if __name__=='__main__':main()
