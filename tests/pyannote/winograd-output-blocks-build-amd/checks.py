"""Permit only the two-output-block MultiplyWinograd512 change."""


def inventory(value, measured, built):
    assert value['inventory_complete'] and len(value['observations']) == 2
    changed = None
    for row, (name, count) in zip(value['observations'], [('Lokad.Onnx.dll',3179), ('Lokad.Onnx.Data.dll',697)]):
        assert row['assembly'] == name and row['methods'] == count
        assert row['public_surface_equal'] and row['compiler_rename'] is None
        assert not row['added'] and not row['removed']
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert len(row['normalized_methods']) == count
        if name == 'Lokad.Onnx.dll':
            changed, = row['differences']
            assert changed.startswith('Lokad.Onnx.ConvBlockedSpatial::MultiplyWinograd512::')
            assert set(row['candidate_methods']) == {changed}
            assert row['normalized_methods'][changed] != row['candidate_methods'][changed]
            assert row['unchanged_methods'] == count-1
        else:
            assert not row['differences'] and not row['candidate_methods']
            assert row['unchanged_methods'] == count
    return dict(passed=True, core_methods=3179, unchanged_core_methods=3178, data_methods=697,
                changed=changed, public_surface_equal=True)
