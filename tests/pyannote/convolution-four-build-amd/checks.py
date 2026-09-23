"""Permit only Execute dispatch and a new private four-block kernel."""


def inventory(value,measured,built):
    assert value['inventory_complete'] and len(value['observations'])==2
    changed=added=None
    for row,(name,count) in zip(value['observations'],[('Lokad.Onnx.dll',3163),('Lokad.Onnx.Data.dll',697)]):
        assert row['assembly']==name and row['methods']==count
        assert row['public_surface_equal'] and row['compiler_rename'] is None
        assert not row['removed']
        assert row['before_sha256']==measured[name]['sha256'] and row['after_sha256']==built[name]['sha256']
        assert len(row['normalized_methods'])==count
        if name=='Lokad.Onnx.dll':
            changed,=row['differences'];added,=row['added']
            assert changed.startswith('Lokad.Onnx.ConvBlockedSpatial::Execute::')
            assert added.startswith('Lokad.Onnx.ConvBlockedSpatial::Kernel512Four::')
            assert set(row['candidate_methods'])=={changed,added}
            assert row['normalized_methods'][changed]!=row['candidate_methods'][changed]
            assert added not in row['normalized_methods'] and row['candidate_methods'][added]
            assert row['unchanged_methods']==count-1
        else:
            assert not row['added'] and not row['differences'] and not row['candidate_methods']
            assert row['unchanged_methods']==count
    return dict(passed=True,existing_core_methods=3163,candidate_core_methods=3164,
        unchanged_core_methods=3162,data_methods=697,changed=changed,added=added,public_surface_equal=True)
