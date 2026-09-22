"""Verify both successor archives and the complete unchanged product inventory."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'tests/pyannote/combined-amd-v2'))
from transport import BASE, PREPARED, checked_local
from candidate_protocol import pin, read, write, verified_files
from resume_prefix import retained


def archive_check(path, files):
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        assert len(members) == len(files) and {m.name for m in members} == set(files)
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            assert member.size == files[member.name]['bytes']
            with archive.extractfile(member) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == files[member.name]['sha256']
    return len(files)


def main():
    prepared, bundle, execution = checked_local()
    payload = PREPARED / 'payload'
    spec = read(payload / 'payload.json')
    old_root = ROOT / 'artifacts/pyannote-combined-amd-payload-20260922/payload'
    old = read(old_root / 'payload.json')
    verified_files(payload, spec['files'])
    verified_files(old_root, old['files'])
    for key in ['cores', 'source_files', 'external', 'protocol', 'role_labels', 'combined_qualification', 'portable_qualification']:
        assert spec[key] == old[key], key
    differences = {name for name, wanted in old['files'].items() if pin(payload / name) != wanted}
    assert differences == {'runtimes/portable/GraphQualification.dll', 'runtimes/rows/GraphQualification.dll'}
    corrected = ROOT / 'artifacts/pyannote-combined-consumers-20260922'
    assert pin(corrected / 'closed.json')['sha256'] == '602d752956d6cf830851cd0dc35eb4bdf1230880fb2dce6d8bbea6f530bb555c'
    for name, wanted in read(corrected / 'closed.json')['files'].items():
        assert pin(ROOT / name) == wanted
    for role in ['portable', 'rows']:
        assert pin(payload / 'runtimes' / role / 'GraphQualification.dll') == pin(corrected / role / 'bin/Release/net10.0/GraphQualification.dll')
    collected, collection, state = retained(payload)
    verified_files(payload, read(collected / 'campaign/built-files.json'))
    assert len(state['runs'][:-1]) == 14 and all(r['code'] == 0 for r in state['runs'][:-1])
    assert 'Ran 21 tests' in (BASE / 'selftest.log').read_text() and '\nOK\n' in (BASE / 'selftest.log').read_text()
    payload_count = archive_check(PREPARED / 'payload.tar.gz', dict(spec['files'], **{'payload.json': prepared['payload']}))
    execution_count = archive_check(BASE / 'execution.tar.gz', dict(execution['files'], **{'execution.json': bundle['execution']}))
    receipt = dict(passed=True, prepared=pin(PREPARED / 'prepared.json'), bundle=pin(BASE / 'prepared.json'),
        payload_archive=prepared['archive'], execution_archive=bundle['archive'], payload_members=payload_count,
        execution_members=execution_count, product_changes=0, consumer_changes=sorted(differences),
        numerical_and_timing_gates_unchanged=True, reused_successful_stages=14, selftests=21,
        verifier=pin(Path(__file__).resolve()), amd_deployed=False)
    write(BASE / 'preparation-verified.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
