"""Confine the candidate to declared packing/dispatch and copied arithmetic."""

def inventory(value,measured,built):
    assert value['inventory_complete'] and len(value['observations'])==2
    reviewed=[]
    expected_added={
        'Lokad.Onnx.ConvBlockedSpatial::'+name for name in ['PrepareWinograd','PlanWinograd','ExecuteWinograd','EpilogueRange',
            'TransformWinogradInput','TransformWinogradInputContiguous','MultiplyWinograd256','MultiplyWinograd512','OutputWinograd256','OutputWinograd512']}
    expected_added|={'Lokad.Onnx.GraphConvPacking::'+name for name in ['WinogradShape','WinogradConsumers','ResolveRecord']}
    expected_added|={'Lokad.Onnx.PackedConvWeight::'+name for name in ['get_WinogradValues','set_WinogradValues']}
    expected_added.add('Lokad.Onnx.Tensor`1[T]::TryConvWinograd')
    mandatory={'Lokad.Onnx.GraphConvPacking::'+name for name in ['PruneAndBytes','PackWeights','Resolve']}
    mandatory.add('Lokad.Onnx.Tensor`1[T]::TryConvBlockedSpatial')
    for row,(name,count) in zip(value['observations'],[('Lokad.Onnx.dll',3163),('Lokad.Onnx.Data.dll',697)]):
        assert row['assembly']==name and row['methods']==count
        assert row['public_surface_equal'] and row['compiler_rename'] is None and not row['removed']
        assert row['before_sha256']==measured[name]['sha256'] and row['after_sha256']==built[name]['sha256']
        assert len(row['normalized_methods'])==count
        changes={k.rsplit('::',1)[0] for k in row['differences']}
        additions={k.rsplit('::',1)[0] for k in row['added']}
        assert len(additions)==len(row['added'])
        assert set(row['candidate_methods'])==set(row['differences'])|set(row['added'])
        if name=='Lokad.Onnx.dll':
            assert additions==expected_added,dict(missing=list(expected_added-additions),extra=list(additions-expected_added))
            assert mandatory<=changes
            assert all(k in mandatory or k.startswith('Lokad.Onnx.PackedConvWeight::') for k in changes),changes
        else:assert not additions and not changes
        assert row['unchanged_methods']==count-len(row['differences'])
        for key in row['differences']:assert row['normalized_methods'][key]!=row['candidate_methods'][key]
        reviewed.append(dict(assembly=name,original_methods=count,unchanged_methods=row['unchanged_methods'],
            added=row['added'],changed=row['differences'],public_surface_equal=True))
    return dict(passed=True,assemblies=reviewed,source_scope='Only optional prepared weights, dispatch and admitted Winograd methods.')
