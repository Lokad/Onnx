"""Bind unchanged shared consumers to the admitted attention ownership pair."""


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
        ('Lokad.Onnx.ComputationalGraph', 'PrepareOwnedMatMulWeights')}
    assert compatible['selected'] == models['identities']['selected']
    assert compatible['candidate'] == models['identities']['candidate']
    assert compatible['selected']['Lokad.Onnx.dll']['sha256'] == 'a6f7d9f9abf0dc3c10a1b566443a22352c8cc84f05a198766325fef2b09850a4'
    assert compatible['candidate']['Lokad.Onnx.dll']['sha256'] == 'ee5218dbab0a970b0f20f273fc2a839bb9b96e9d5438791529eb11c731d66859'
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
    assert compatible['focused_contracts']['sha256'] == 'a293403a5dc3a7bdabb7fdc448470feea2b54dbf139d98b7b3f9ed0eef39854f'
    assert compatible['actual_model_census']['sha256'] == 'ef4e41eca49d33f6cac4ae7d510922053874002787709cc3d83bc98c3437d3d7'
    assert compatible['selected']['Lokad.Onnx.Data.dll'] == compatible['candidate']['Lokad.Onnx.Data.dll']
    assert not models.get('release_admitted', False)
