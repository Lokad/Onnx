"""Independent archive, candidate and prospective protocol verification."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / 'tests/pyannote/combined-amd'
sys.path.insert(0, str(TOOLS))
from transport import BASE, PREPARED, checked_local
from candidate_protocol import pin, read, write, verified_files, ROLES, TIMING_ROLES, LIMITS
from admission import ROLE_LABELS


def archive_check(path, files):
    with tarfile.open(path) as archive:
        members = archive.getmembers()
        assert len(members) == len(files) and {m.name for m in members} == set(files)
        for member in members:
            assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            expected = files[member.name]
            assert member.size == expected['bytes']
            with archive.extractfile(member) as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == expected['sha256']
    return len(files)


def main():
    assert not (BASE / 'preparation-verified.json').exists()
    prepared, bundle, execution = checked_local()
    payload = PREPARED / 'payload'
    spec = read(payload / 'payload.json')
    verified_files(payload, spec['files'])
    assert ROLES == ('production', 'portable', 'rows')
    assert TIMING_ROLES == ('production', 'portable', 'rows', 'ort', 'ort', 'rows', 'portable', 'production')
    assert spec['role_labels'] == ROLE_LABELS and execution['limits'] == LIMITS
    assert spec['protocol']['natural_meetings'] == 3
    assert spec['protocol']['performance_admission'] == pin(TOOLS / 'admission.py')
    assert spec['protocol']['total_timing_calls'] == 128 and spec['protocol']['measured_timing_calls'] == 96
    assert spec['combined_qualification'] == pin(ROOT / 'artifacts/pyannote-combined-avx512-20260922/closed.json')
    assert spec['portable_qualification'] == pin(ROOT / 'artifacts/pyannote-portable-applications-20260922/closed.json')
    expected = {
        'production': ('29477d505dd230aef0b5aa2792da8ec76c6903c5d2cff244327d803b432b4cbb', 'e7fe1668e3aa08fb07b1e5a687ef2b1e4af54567f6a458db09d411eb69f99aeb'),
        'portable': ('e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838', '85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'),
        'rows': ('e36963d848ff907acc521908c3a82569dd096c9e0e9ccad864a0d26028082d80', '2b512f2527d3ede6e9a50b2206aabcb6bd4595c96c0ebcecdebe9e7a21939393')}
    for role, (core, data) in expected.items():
        runtime = payload / 'runtimes' / role
        assert pin(runtime / 'Lokad.Onnx.dll')['sha256'] == core
        assert pin(runtime / 'Lokad.Onnx.Data.dll')['sha256'] == data
        for family in ['pyannote', 'parakeet']:
            manifest = read(payload / 'manifests' / (role + '-' + family + '.json'))
            assert manifest['core_sha256'] == core and manifest['data_sha256'] == data
            assert manifest['product_source'] == ROLE_LABELS[role]
        for name in ['AudioBenchmark', 'GraphQualification', 'TranscribeReplay', 'NaturalMeetings']:
            for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
                assert pin(runtime / (name + '.' + suffix)) == pin(payload / 'runtimes/rows' / (name + '.' + suffix))
    assert pin(payload / 'runtimes/rows/AudioBenchmark.dll')['sha256'] == '7eca033a1b986a4cb90621392639d230c95097cb703dd25274fd72d66c5ba4f1'
    assert pin(payload / 'runtimes/rows/NaturalMeetings.dll')['sha256'] == '79e3e7990ba6aa29e42da788277aad41b774ff3b8c3966b18ab1101944d0c0f1'
    actual_sources = {p.relative_to(payload / 'source').as_posix(): pin(p) for p in (payload / 'source').rglob('*') if p.is_file()}
    assert actual_sources == spec['source_files']
    original_source = ROOT / 'artifacts/pyannote-combined-avx512-20260922/source'
    for name, wanted in actual_sources.items():
        assert pin(original_source / name) == wanted
    meetings = read(payload / 'meetings/manifest.json')
    assert meetings['core_sha256'] == expected['rows'][0] and meetings['data_sha256'] == expected['rows'][1]
    assert [c['samples'] for c in meetings['cases']] == [9600000, 9600000, 480000]
    assert 'Ran 19 tests' in (BASE / 'selftest.log').read_text() and '\nOK\n' in (BASE / 'selftest.log').read_text()
    payload_members = archive_check(PREPARED / 'payload.tar.gz', dict(spec['files'], **{'payload.json': prepared['payload']}))
    execution_members = archive_check(BASE / 'execution.tar.gz', dict(execution['files'], **{'execution.json': bundle['execution']}))
    receipt = dict(passed=True, prepared=pin(PREPARED / 'prepared.json'), bundle=pin(BASE / 'prepared.json'),
        payload_archive=prepared['archive'], execution_archive=bundle['archive'], role_labels=ROLE_LABELS,
        sources=len(actual_sources), payload_members=payload_members, execution_members=execution_members,
        selftests=19, verifier=pin(Path(__file__).resolve()), amd_deployed=False)
    write(BASE / 'preparation-verified.json', receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
