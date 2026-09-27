"""Refuse any compiled product change outside the selected reader and dispatch."""
def review(value, built, baseline):
    assert value['inventory_complete'] and len(value['observations']) == 2
    results = []
    for row in value['observations']:
        core = row['assembly'] == 'Lokad.Onnx.dll'
        assert core or row['assembly'] == 'Lokad.Onnx.Data.dll'
        assert row['before_sha256'] == baseline[row['assembly']]['sha256']
        wanted = built['products']['candidate'] if core else baseline[row['assembly']]
        assert row['after_sha256'] == wanted['sha256']
        assert row['methods'] == (3283 if core else 697)
        assert row['public_surface_equal'] and row['public_surface'] == row['public_surface_after']
        assert row['assembly_attributes_before'] == row['assembly_attributes_after']
        assert not row['removed']
        assert {k: row['method_flags_after'][k] for k in row['method_flags_before']} == row['method_flags_before']
        if core:
            assert len(row['differences']) == 2 and {n.split('::')[1] for n in row['differences']} == {'ResolvePackedKernel', 'RunPreparedPackedRows'}
            assert all(n.startswith('Lokad.Onnx.Tensor`1[T]::') for n in row['differences'])
            assert len(row['added']) == 1 and row['added'][0].startswith('Lokad.Onnx.PreparedSingleRowKernel::Multiply::')
            assert row['unchanged_methods'] == 3281
        else:
            assert not row['differences'] and not row['added'] and row['unchanged_methods'] == 697
        results.append(dict(assembly=row['assembly'], unchanged=row['unchanged_methods'], changed=row['differences'], added=row['added']))
    return dict(passed=True, assemblies=results, public_surface_equal=True, flags_equal=True, assembly_metadata_equal=True)
