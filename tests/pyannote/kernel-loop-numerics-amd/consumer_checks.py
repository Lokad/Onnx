"""Allow only the embedded product hash to change in each qualified consumer."""
OLD = '3ca0a2a5ff1b129272da05b679d539d865a3c6c20ee8625c24955e2a5f497c6a'


def consumer_inventory(value, mode, previous, current, core):
    assert value['inventory_complete']
    row, = value['observations']
    assembly = 'LayerGraphs.dll' if mode == 'layers' else 'Lokad.Onnx.Backend.Tests.dll'
    assert row['assembly'] == assembly and row['public_surface_equal']
    assert not row['added'] and not row['removed'] and row.get('compiler_rename') is None
    count = 134 if mode == 'layers' else 145
    assert row['methods'] == len(row['normalized_methods']) == count and row['unchanged_methods'] == count-1
    key, = row['differences']
    assert key == ('ModelProbe' if mode == 'layers' else 'Probe')+'::Main::Int32 Main(System.String[])'
    assert set(row['candidate_methods']) == {key}
    before, after = row['normalized_methods'][key], row['candidate_methods'][key]
    assert before.count(OLD) == 1 and before.replace(OLD,core) == after
    assert row['before_sha256'] == previous['sha256'] and row['after_sha256'] == current['sha256']
    return dict(passed=True, mode=mode, methods=count, unchanged=count-1, changed=[key], only_core_literal_changed=True)
