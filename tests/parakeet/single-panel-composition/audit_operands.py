"""Verify transitive, previously closed operand fixtures used by the unchanged probe."""
from pathlib import Path
from prepare import ROOT, BASE, CONTROL, pin, read, save, verify, terminal, rel


def main():
    target = BASE / 'external-operand-proof.json'
    assert not target.exists() and not (BASE / 'closed.json').exists()
    state = read(BASE / 'processes.json')
    assert state['complete'] and state['code'] == 0
    terminal(state['supervisor'])
    folders = [('parakeet-reduction-accuracy-20260921', 'cea91b423bfbb8da0059617dceeebb3f4175256480576abc925b9bdec5324629'),
        ('parakeet-reduction-model-20260921', '71f833ec9260a26d38456ad641a3a436ea215ac44f70b760aabaef826e128742')]
    files = {rel(Path(__file__).resolve()): pin(Path(__file__).resolve())}
    original = {}
    for folder, sha in folders:
        path = ROOT / 'artifacts' / folder / 'closed.json'
        assert pin(path)['sha256'] == sha
        proof = read(path)
        assert proof.get('passed', proof.get('native_numeric_passed'))
        verify(proof['files'])
        original.update(proof['files'])
        files[rel(path)] = pin(path)
    accuracy = ROOT / 'artifacts/parakeet-reduction-accuracy-20260921'
    operands = [accuracy / 'weight.bin']
    for route in ('managed-native', 'native-native', 'managed-managed', 'native-managed'):
        operands += [ROOT / 'artifacts/parakeet-projection-20260921/outputs' / route / '03.bin',
            accuracy / 'output' / route / '256-projection.bin']
    operands += list((ROOT / 'artifacts/parakeet-reduction-model-20260921/runtime').glob('*.dll'))
    for path in operands:
        assert pin(path) == original[rel(path)]
        files[rel(path)] = pin(path)
    normal, disabled = read(BASE / 'geometry.json'), read(BASE / 'hardware-off.json')
    assert normal['passed'] and disabled['passed']
    assert (normal['tests'], normal['eligible'], normal['fallback'], normal['changed'], normal['raw_contracts'],
        normal['dynamic_dispatch_cases'], normal['actual_operand_controls']) == (416, 64, 352, 64, 416, 48, 4)
    assert disabled['tests'] == 4
    for value in (normal, disabled):
        assert value['core_sha256'] == pin(BASE / 'runtime/Lokad.Onnx.dll')['sha256']
        assert value['original_core_sha256'] == pin(CONTROL / 'Lokad.Onnx.dll')['sha256']
    save(target, dict(passed=True, files=files, geometry=pin(BASE / 'geometry.json'),
        hardware_off=pin(BASE / 'hardware-off.json'), operand_files=len(operands),
        scope='Post-run verification against previously closed exact fixture and prior-runtime identities; no new probe execution or model qualification.'))
    print(dict(passed=True, operand_files=len(operands), proof=pin(target)))


if __name__ == '__main__':
    main()
