"""Retire completed graph VM arrays while retaining every verified local output."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'artifacts/parakeet-decoder-pyannote-app-headroom-20260927'
BASE = ROOT/'artifacts/parakeet-decoder-packed-row-graphs-amd-20260927'
REMOTE_BASE = '/dev/shm/lokad-parakeet-decoder-packed-row-graphs-20260927'
REUSED = ROOT/'eng/retire_decoder_graph_headroom.py'
loader = importlib.util.spec_from_file_location('closed_output_retirement', REUSED)
reused = importlib.util.module_from_spec(loader)
loader.loader.exec_module(reused)
pin, read = reused.pin, reused.read


def replacement(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    assert not OUT.exists(), 'Preserve any completed or partial retirement'
    assert pin(REUSED)['sha256'] == 'b31bf3daef1e3bcf1707622182a7701e090451da14eecea88b913c21c6690daa'
    assert pin(BASE/'closed.json')['sha256'] == '485769614ee3f0d5becc50ca107a07893e61eb495f162d3301614e53bbe09108'
    proof = read(BASE/'closed.json')
    receipt = read(BASE/'collected/collection.json')
    initial = read(BASE/'payload.json')
    assert proof['passed'] and proof['admitted'] and proof['all_controls_passed']
    for name in ['collected/collection.json', 'payload.json']:
        assert pin(BASE/name) == proof['files'][name]
    assert receipt['terminal'] and receipt['code'] == 0 and receipt['input_error'] is None
    assert len(initial['jobs']) == 72 and len(set(initial['jobs'])) == 72
    rows = []
    for name, wanted in receipt['files'].items():
        parts = Path(name).parts
        if len(parts) != 3 or parts[0] not in initial['jobs'] or parts[1] != 'output' or Path(name).suffix != '.f32':
            continue
        assert name not in initial['files']
        assert pin(BASE/'collected'/name) == wanted == proof['files']['collected/'+name]
        rows.append(dict(name=name, identity=wanted))
    assert len(rows) == 297
    expected = sum(((r['identity']['bytes']+4095)//4096)*4096 for r in rows)
    assert expected == 23998464
    group = dict(local=BASE.name, base=REMOTE_BASE, kind='graphs', roles=initial['jobs'],
                 files=rows, closure=pin(BASE/'closed.json'), receipt=pin(BASE/'collected/collection.json'))
    script = replacement(reused.REMOTE, '@@GROUPS@@', repr([group]))
    script = replacement(script,
        "roles=['selected','candidate'] if group['kind']=='pyannote' else ['selected-shared','selected-e5','candidate-shared','candidate-e5']",
        "assert group['kind']=='graphs';roles=group['roles'];assert len(roles)==len(set(roles))==72")
    script = replacement(script, 'assert len(paths)==428 and physical==98713600',
                         'assert len(paths)==297 and physical==23998464')
    compile(script, 'retire-completed-graph-duplicates', 'exec')
    OUT.mkdir()
    def save(name, value):
        with (OUT/name).open('x', encoding='utf8', newline='\n') as stream:
            json.dump(value, stream, indent=2); stream.write('\n')
    save('intent.json', dict(tool=pin(Path(__file__)), reused_safety_checks=pin(REUSED),
                            groups=[group], expected_physical_bytes=expected))
    with (OUT/'remote.py').open('x', encoding='utf8', newline='\n') as stream:
        stream.write(script)
    transport_loader = importlib.util.spec_from_file_location('retirement_transport', ROOT/'tests/parakeet/ort-diagnosis-amd/run.py')
    transport = importlib.util.module_from_spec(transport_loader)
    transport_loader.loader.exec_module(transport)
    result = transport.ssh(script)
    save('result.json', result)
    assert result['passed'] and result['files'] == 297 and result['physical_bytes_freed'] == expected
    for row in rows:
        assert pin(BASE/'collected'/row['name']) == row['identity']
    save('closed.json', dict(passed=True, result=pin(OUT/'result.json'), intent=pin(OUT/'intent.json'),
                            script=pin(OUT/'remote.py'), locally_retained_outputs=297))
    print(json.dumps(dict(**result, closure=pin(OUT/'closed.json'))))


if __name__ == '__main__':
    main()
