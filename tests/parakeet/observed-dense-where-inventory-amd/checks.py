"""Match the current release everywhere except the exact retained selector methods."""
from il_body import normalized_body

PROVIDER = 'Lokad.Onnx.CPUExecutionProvider::Where::Lokad.Onnx.OpResult Where(Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ExecutionOptions)'
TRY = 'Lokad.Onnx.DenseScalarWhere::Try::Boolean Try[T](Lokad.Onnx.Tensor`1[System.Boolean], Lokad.Onnx.Tensor`1[T], Lokad.Onnx.Tensor`1[T], Lokad.Onnx.Tensor`1[T] ByRef)'
MIXED = 'Lokad.Onnx.DenseScalarWhere::SelectMixed::Void SelectMixed[T](System.ReadOnlySpan`1[System.Byte], T, System.ReadOnlySpan`1[T], System.Span`1[T], System.ReadOnlySpan`1[System.Int32], System.ReadOnlySpan`1[System.Int32], System.ReadOnlySpan`1[System.Int32])'
ADDED = {TRY: 520, MIXED: 520}


def inventory(value, measured, built, prior):
    assert prior['passed'] and value['inventory_complete'] and len(value['observations']) == 2
    assert set(prior['retained_methods']) == {PROVIDER, *ADDED}
    for row, (name, count) in zip(value['observations'], [('Lokad.Onnx.dll', 3251), ('Lokad.Onnx.Data.dll', 697)], strict=True):
        assert row['assembly'] == name and row['methods'] == len(row['normalized_methods']) == count
        assert row['before_sha256'] == measured[name]['sha256'] and row['after_sha256'] == built[name]['sha256']
        assert row['public_surface_equal'] and row['compiler_rename'] is None and not row['removed']
        before, after = row['method_flags_before'], row['method_flags_after']
        assert set(before) == set(row['normalized_methods'])
        assert row['normalized_methods'] == prior['current_methods'][name]
        assert before == prior['current_flags'][name]
        if name == 'Lokad.Onnx.Data.dll':
            assert row['unchanged_methods'] == 697 and not row['differences'] and not row['added']
            assert not row['candidate_methods'] and before == after
            continue
        assert row['differences'] == [PROVIDER] and row['unchanged_methods'] == 3250
        assert set(row['added']) == set(ADDED) and after == before | ADDED
        assert set(row['candidate_methods']) == {PROVIDER, *ADDED}
        for key, body in prior['retained_methods'].items():
            assert normalized_body(row['candidate_methods'][key]) == normalized_body(body), key
            assert after[key] == prior['retained_flags'][key], key
    return dict(passed=True, original_core_methods=3251, candidate_core_methods=3253,
        unchanged_core_methods=3250, data_methods=697, changed_method=PROVIDER,
        retained_provider_and_helper_bodies_exact=True, generic_tensor_where_exact=True,
        all_existing_flags_exact=True, added_flags=ADDED, public_surface_equal=True,
        new_composition_numerically_qualified=False)
