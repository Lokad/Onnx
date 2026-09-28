"""Bind unchanged shared consumers to the admitted sigmoid pair."""


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
        ('Lokad.Onnx.CPUExecutionProvider', 'Sigmoid')}
    assert {tuple(n.split('::')[:2]) for n in core['added']} == {
        ('Lokad.Onnx.CPUExecutionProvider','SigmoidRationalAvx512'),
        ('Lokad.Onnx.CPUExecutionProvider','SigmoidRational512')}
    assert len(core['added']) == 2 and not data['added']
    assert compatible['selected'] == models['identities']['selected']
    assert compatible['candidate'] == models['identities']['candidate']
    assert compatible['selected']['Lokad.Onnx.dll']['sha256'] == 'e98edee2c62a27e8a7771c49a1381c74b0c826f6e235f6e52b2ee4385c76efcc'
    assert compatible['candidate']['Lokad.Onnx.dll']['sha256'] == 'bdcfe20d3e27a964d0070c51faefe6c10936c8150bf0e8ac3ce1bde5a6dc2efb'
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
    assert compatible['focused_contracts']['sha256'] == '0953618fa2191a042db85eaa1b2a0b484f379f8f38b5879f102be2c3818bef15'
    assert compatible['selected']['Lokad.Onnx.Data.dll'] == compatible['candidate']['Lokad.Onnx.Data.dll']
    assert not models.get('release_admitted', False)
