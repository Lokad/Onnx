"""Preserve the first compiler-ordinal isolation failure before a successor."""
import json
from common import *


def main():
    assert not (BASE / 'failure-closed.json').exists()
    state = read(BASE / 'preparation.json')
    assert state['complete'] and state['code'] == 1
    assert [r['name'] for r in state['runs']] == ['core-restore', 'core-build', 'bridge-restore', 'bridge-build', 'instructions']
    terminal(state['supervisor'])
    identities = [state['supervisor']]
    samples = 0
    for run in state['runs']:
        assert run['complete'] and (run['code'] != 0 if run['name'] == 'instructions' else run['code'] == 0)
        assert run['preflight']['available'] >= 8 * 1024**3
        for pid, birth in run['members'].items():
            identity = dict(pid=int(pid), birth=birth)
            terminal(identity)
            identities.append(identity)
        rows = [json.loads(line) for line in (BASE / 'logs' / (run['name'] + '.samples.jsonl')).read_text().splitlines()]
        assert len(rows) == run['samples'] > 0 and max(r['rss'] for r in rows) == run['peak_rss']
        for row in rows:
            assert row['seconds'] < 900 and row['rss'] < 8 * 1024**3 and row['available'] >= 1024**3 and row['disk'] >= 20 * 1024**3
            assert row['output_bytes'] <= 1024**3 and row['rss'] == sum(p['rss'] for p in row['members'])
            assert all(p['affinity'] == [2] and run['members'][str(p['pid'])] == p['birth'] for p in row['members'])
        samples += len(rows)
    line = (BASE / 'logs/instructions.log').read_text().splitlines()[0]
    assert line.startswith('Unhandled exception. System.IO.InvalidDataException: ')
    inventory = json.loads(line.split(': ', 1)[1])
    assert inventory['name'] == 'Lokad.Onnx.dll' and inventory['removed'] and inventory['added'] and inventory['differences']
    save(BASE / 'failed-instructions.json', inventory)
    files = {}
    for folder in [BASE, TOOLS]:
        for path in folder.rglob('*'):
            if path.is_file() and not {'obj', 'packages'}.intersection(path.relative_to(folder).parts):
                files[rel(path)] = pin(path)
    files[rel(MONITOR)] = pin(MONITOR)
    save(BASE / 'failure-closed.json', dict(passed=False, stage='instructions', reason='Added methods shift compiler-generated ordinals in partial classes; strict method isolation rejected.',
        files=files, identities=identities, resource_samples=samples, inference_started=False))
    print(json.dumps(dict(closed=pin(BASE / 'failure-closed.json'), samples=samples, identities=len(identities))))


if __name__ == '__main__':
    main()
