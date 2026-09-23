"""Bound the diagnostic consumer change; product DLLs are unchanged."""
def inventory(value,previous,built,product):
    assert value['inventory_complete']
    row,=value['observations']
    assert row['assembly']=='SampledAudio.dll' and row['methods']==160 and row['unchanged_methods']==159
    assert row['public_surface_equal'] and not row['removed'] and row.get('compiler_rename') is None
    assert row['before_sha256']==previous['sha256'] and row['after_sha256']==built['sha256']
    assert len(row['normalized_methods'])==160
    key,=row['differences'];assert key.startswith('Program::<Main>$::')
    added=row['added'];assert len(added)==2
    assert sorted(k.split('::')[1] for k in added)==['FullParakeet','WarmupParakeet']
    assert all(k.startswith('SampledRequests::') for k in added)
    assert set(row['candidate_methods'])=={key,*added}
    for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']:
        digest=product[name]['sha256'];assert row['normalized_methods'][key].count(digest)==row['candidate_methods'][key].count(digest)==1
    return dict(passed=True,methods=160,unchanged=159,changed=[key],added=added,public_surface_equal=True,
        scope='Only Main family guard and Parakeet wrapper dispatch differ; two wrappers added. Review all changed instructions and source patch before profiling.')
