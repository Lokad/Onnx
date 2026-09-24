"""Admit only the two new selected-product hash literals, with every method flag exact."""
import json

OLD = {
    'Lokad.Onnx.dll': '521bae1702849ca23dda586515e7cbabaac2d1eabdff04dc90a7ba76059e93fb',
    'Lokad.Onnx.Data.dll': 'f3b9aa81ee9766797e95216714dec559c5d8b020f8df510cf5f7ee0dda82693a'}
CURRENT = {
    'Lokad.Onnx.dll': '672e5f303b011e27bb23097a49252c38ddee334938c75895e3c2341df0f3be35',
    'Lokad.Onnx.Data.dll': '065b7a7f28561a37174f9d38ac13ef77bc59c010702e2b64200e9542376eb4c5'}


def inventory(value, previous, built, product):
    assert value['inventory_complete']
    row, = value['observations']
    assert row['assembly'] == 'SampledAudio.dll'
    assert row['methods'] == len(row['normalized_methods']) == 162
    assert row['unchanged_methods'] == 161
    assert row['public_surface_equal'] and not row['removed'] and not row['added']
    assert row.get('compiler_rename') is None
    assert row['before_sha256'] == previous['sha256'] and row['after_sha256'] == built['sha256']
    key, = row['differences']; assert key.startswith('Program::<Main>$::')
    assert set(row['candidate_methods']) == {key}
    assert row['method_flags_before'] == row['method_flags_after']
    assert set(row['method_flags_before']) == set(row['normalized_methods'])
    for marker in ['WarmupParakeet', 'FullParakeet']:
        name, = [k for k in row['method_flags_before'] if k.startswith('SampledRequests::'+marker+'::')]
        assert row['method_flags_before'][name] == 8  # NoInlining, no optimization override.
    before = json.loads(row['normalized_methods'][key])
    after = json.loads(row['candidate_methods'][key])
    offsets = []
    for name, old in OLD.items():
        assert product[name]['sha256'] == CURRENT[name]
        matches = [r for r in before['instructions'] if r['opcode'] == 'ldstr' and r['operand'] == old]
        instruction, = matches
        offsets.append(instruction['offset']); instruction['operand'] = CURRENT[name]
    assert before == after, 'Unexpected Main instruction, local, branch, stack or exception change'
    return dict(passed=True, methods=162, unchanged=161, changed=[key], added=[],
        public_surface_equal=True, implementation_flags_equal=True, changed_hash_literals=2,
        main_instructions=len(before['instructions']), changed_offsets=offsets,
        scope='Only the two selected-product hash operands change; all other compiled instructions and implementation flags match.')
