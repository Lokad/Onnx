"""Require a short wrapper, two private additions and the exact old general body."""


def inventory(value, measured, built):
    assert value['inventory_complete'] and len(value['observations']) == 2
    changed = general = short = None
    for row, (name, count) in zip(value['observations'], [('Lokad.Onnx.dll',3179), ('Lokad.Onnx.Data.dll',697)]):
        assert row['assembly'] == name and row['methods'] == count
        assert row['public_surface_equal'] and row['compiler_rename'] is None
        assert not row['removed']
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert len(row['normalized_methods']) == count
        if name == 'Lokad.Onnx.dll':
            changed, = row['differences']
            assert changed.startswith('Lokad.Onnx.Tensor`1[T]::RunFloatMatMulKernel::')
            signature = changed.split('::', 2)[2]
            general = 'Lokad.Onnx.Tensor`1[T]::RunGeneralFloatMatMulKernel::'+signature.replace('RunFloatMatMulKernel(', 'RunGeneralFloatMatMulKernel(')
            short = 'Lokad.Onnx.Tensor`1[T]::RunShortWidePackedRows::'+signature.replace('RunFloatMatMulKernel(', 'RunShortWidePackedRows(')
            assert len(row['added']) == 2 and set(row['added']) == {general, short}
            assert set(row['candidate_methods']) == {changed, general, short}
            assert row['normalized_methods'][changed] != row['candidate_methods'][changed]
            assert row['normalized_methods'][changed] == row['candidate_methods'][general]
            assert row['candidate_methods'][short]
            assert row['unchanged_methods'] == count-1
        else:
            assert not row['added'] and not row['differences'] and not row['candidate_methods']
            assert row['unchanged_methods'] == count
    return dict(passed=True, original_core_methods=3179, candidate_core_methods=3181,
        unchanged_core_methods=3178, data_methods=697, changed=changed,
        added=[general, short], original_general_body_exact=True, public_surface_equal=True)
