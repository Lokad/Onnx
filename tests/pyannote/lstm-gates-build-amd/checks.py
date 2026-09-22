"""Constrain the complete product inventory to the declared M26 experiment."""
CHANGED={('Lokad.Onnx.CPUExecutionProvider','Lstm')}
ADDED={('Lokad.Onnx.CPUExecutionProvider','LstmUpdateDefaultGates')}


def names(keys):
    value=[tuple(key.split('::')[:2]) for key in keys]
    assert len(value)==len(set(value))
    return set(value)


def inventory(value,measured,built):
    assert value['inventory_complete'] and len(value['observations'])==2
    changes=None;added=None
    for row,(name,count) in zip(value['observations'],[('Lokad.Onnx.dll',3163),('Lokad.Onnx.Data.dll',697)]):
        assert row['assembly']==name and row['methods']==count and len(row['normalized_methods'])==count
        assert row['public_surface_equal'] and row['compiler_rename'] is None and not row['removed']
        assert row['before_sha256']==measured[name]['sha256'] and row['after_sha256']==built[name]['sha256']
        if name=='Lokad.Onnx.dll':
            changes=row['differences'];added=row['added']
            assert names(changes)==CHANGED and names(added)==ADDED and row['unchanged_methods']==3162
            assert set(row['candidate_methods'])==set(changes)|set(added)
            assert all(key in row['normalized_methods'] and row['normalized_methods'][key]!=row['candidate_methods'][key] for key in changes)
            assert all(key not in row['normalized_methods'] and row['candidate_methods'][key]!='NO-BODY' for key in added)
            assert all('FusedMultiplyAdd' not in row['candidate_methods'][key] for key in added)
        else:
            assert row['unchanged_methods']==697 and not row['differences'] and not row['added'] and not row['candidate_methods']
    return dict(passed=True,existing_core_methods=3163,unchanged_core_methods=3162,added_core_methods=1,
        data_methods=697,changed=changes,added=added,public_surface_equal=True)
