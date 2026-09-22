"""Preserve the failed campaign and derive a terminal-aware monitor successor."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'direct-amd'


def replace(text, old, new, count):
    assert text.count(old) == count, (old, text.count(old), count)
    return text.replace(old, new)


for path in sorted(OLD.iterdir()):
    if path.suffix not in ['.py', '.md'] or path.name in ['derive.py', 'close_failure.py']: continue
    target = HERE / path.name; assert not target.exists()
    text = path.read_text(encoding='utf8')
    if path.name in ['transport.py', 'README.md']:
        text = text.replace('pyannote-direct-amd-payload-20260922', 'pyannote-direct-amd-payload-v2-20260922')
        text = text.replace('pyannote-direct-amd-execution-20260922', 'pyannote-direct-amd-execution-v2-20260922')
        text = text.replace('lokad-pyannote-direct-20260922', 'lokad-pyannote-direct-v2-20260922')
        text = text.replace('tests/pyannote/direct-amd/', 'tests/pyannote/direct-amd-v2/')
    if path.name == 'prepare.py':
        text = replace(text, '    evidence = {}', '''    previous = ROOT / 'artifacts/pyannote-direct-amd-execution-20260922'
    failure = read(previous / 'failure-closed.json')
    assert pin(previous / 'failure-closed.json')['sha256'] == '5bfd1e51e4aa6788e9470be5b8be8d26db424af8cbbeb25fbb0ef8364ae9f8a4'
    assert failure['passed'] and not failure['campaign_passed'] and not failure['timing_started']
    verified_files(previous, failure['files'])
    evidence = {(previous / 'failure-closed.json').relative_to(ROOT).as_posix(): pin(previous / 'failure-closed.json')}''', 1)
    if path.name == 'supervise.py':
        text = replace(text, 'import psutil\n', 'import psutil\nfrom terminal_snapshot import observe as terminal_snapshot\n', 1)
        text = replace(text, '''                    if not members and child.poll() is not None:
                        break
''', '', 1)
        old = "                    stream.write(json.dumps(sample)+'\\n'); stream.flush()"
        new = '''                    transition = terminal_snapshot(child, members, row['members'], absent,
                        sample['seconds'], sample['available'], sample['tmpfs_free'], sample['artifact_bytes'])
                    if transition is not None:
                        row['terminal_transition'] = transition
                        write(folder/'terminal-transition.json', transition)
                        save()
                        break
''' + old
        text = replace(text, old, new, 1)
    if path.name == 'audit_results.py':
        old = "        samples += len(resource)"
        new = '''        if 'terminal_transition' in run:
            from terminal_snapshot import validate
            event = read(campaign/run['name']/'terminal-transition.json')
            validate(event)
            assert event == run['terminal_transition']
            assert event['identities'] == run['members'] and event['code'] == run['code']
            assert event['seconds'] <= run['seconds']
''' + old
        text = replace(text, old, new, 1)
    target.write_text(text, encoding='utf8')
print('Created successor; live-sample gates unchanged, empty samples require confirmed terminal root and descendants.')
