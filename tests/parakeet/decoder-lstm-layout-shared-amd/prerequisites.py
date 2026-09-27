"""Require unchanged consumers, exact compiled ancestry and actual application gain."""


def validate(compatible, models, previous, app, consumer):
    assert compatible['passed'] and compatible['original_public_bindings_preserved']
    assert compatible['all_original_method_flags_preserved'] and compatible['underlying_methods_reconciled'] == 3981
    assert compatible['all_data_methods_exact'] and compatible['no_consumer_or_product_build']
    compiled = compatible['compiled']
    assert compiled['passed'] and compiled['public_surface_equal'] and compiled['flags_equal'] and compiled['attributes_equal']
    core, data = compiled['methods']
    assert core['unchanged'] == 3282 and data['unchanged'] == 697
    assert len(core['changed']) == len(core['added']) == 2 and not data['changed'] and not data['added']
    assert {tuple(n.split('::')[:2]) for n in core['changed']} == {
        ('Lokad.Onnx.CPUExecutionProvider', 'Lstm'), ('Lokad.Onnx.GraphLstmPacking', 'Prepare')}
    assert {tuple(n.split('::')[:2]) for n in core['added']} == {
        ('Lokad.Onnx.PreparedLstmProjection', 'Multiply'), ('Lokad.Onnx.PreparedLstmProjection', 'get_ColumnsPerBlock')}
    assert compatible['selected'] == models['identities']['selected']
    assert compatible['candidate'] == models['identities']['candidate']
    assert compatible['selected']['Lokad.Onnx.dll']['sha256'] == '0d224bcff591563816d64c3a3cc51f7b7cd38f5a1d3c523b05c699a9b82b6b97'
    assert compatible['candidate']['Lokad.Onnx.dll']['sha256'] == 'ad97b4ad632b3306ea3a39d14549ed98540bc8c2e85953a68d89b453d3ed2fdc'
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
    assert not compatible['component_screen_admitted'] and compatible['runtime_diagnosis']['sha256'] == 'dfa4cef574312c04e8d86e6c69151133b2ab4b4f8646bd74fcbe77c12ef23c68'
    assert len(compatible['failed_component_controls']) == 100
    assert all(not r['passed'] for r in compatible['failed_component_controls'])
    assert not models.get('release_admitted', False)
