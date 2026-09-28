"""Bind unchanged shared consumers to the admitted transpose pair."""


def validate(compatible, models, previous, app, consumer):
    for key in ['passed', 'original_public_bindings_preserved', 'all_original_method_flags_preserved',
                'all_data_methods_exact', 'no_consumer_or_product_build']:
        assert compatible[key], key
    assert compatible['underlying_methods_reconciled'] == 3985
    core, data = compatible['compiled_scope']
    assert (core['assembly'], data['assembly']) == ('Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll')
    assert core['unchanged'] == 3287 and data['unchanged'] == 697
    assert len(core['changed']) == 1 and not data['changed']
    assert {tuple(name.split('::')[:2]) for name in core['changed']} == {
        ('Lokad.Onnx.Tensor`1[T]', 'TransposeInto')}
    assert compatible['selected'] == models['identities']['selected']
    assert compatible['candidate'] == models['identities']['candidate']
    assert compatible['selected']['Lokad.Onnx.dll']['sha256'] == '4e97e2ae97e534b99281a361790ac35865b3d8c6171293f36109b24a354eec57'
    assert compatible['candidate']['Lokad.Onnx.dll']['sha256'] == 'c471f5d1ead5889b00b141ce34f2a7689cfe179fbf513da4ce5db84ed01ce277'
    assert compatible['qualified_model_product'] == previous['identities']['candidate']
    assert models['passed'] and previous['passed'] and previous['reference_provenance_verified']
    assert previous['consumer'] == consumer
    assert consumer['sha256'] == 'a50d3e965cf480844559b1f5856e3de318b5c8a269742a9e6504afb9cee6c8c0'
    assert app['passed'] and app['performance']['admitted']
    assert app['identities'] == dict(current=models['identities']['selected'], candidate=models['identities']['candidate'])
    for key, count in [('controls', 63), ('gates', 21)]:
        rows = app['performance'][key]
        assert len(rows) == count and all(row['passed'] for row in rows)
    gate = app['performance']['gates'][-1]
    assert gate['name'] == 'corpus-at-least-one-percent-gain' and gate['limit'] == .99
    assert compatible['focused_contracts']['sha256'] == 'c0c063bb3f73eed2d8d7372d17dd1da3ce4f6a962e190ef101b504fb245e4ecd'
    assert compatible['selected']['Lokad.Onnx.Data.dll'] == compatible['candidate']['Lokad.Onnx.Data.dll']
    assert not models.get('release_admitted', False)
