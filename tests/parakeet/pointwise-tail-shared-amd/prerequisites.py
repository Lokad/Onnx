"""Bind unchanged shared consumers to the admitted pointwise application pair."""


def validate(compatible, models, previous, app, consumer):
    for key in ['passed', 'original_public_bindings_preserved', 'all_original_method_flags_preserved',
                'all_data_methods_exact', 'no_consumer_or_product_build']:
        assert compatible[key], key
    assert compatible['underlying_methods_reconciled'] == 3983
    core, data = compatible['compiled_scope']
    assert (core['assembly'], data['assembly']) == ('Lokad.Onnx.dll', 'Lokad.Onnx.Data.dll')
    assert core['unchanged'] == 3285 and data['unchanged'] == 697
    assert len(core['changed']) == 1 and len(core['added']) == 2
    assert not data['changed'] and not data['added']
    assert {tuple(name.split('::')[:2]) for name in core['changed']} == {
        ('Lokad.Onnx.MathOps', 'mm_unsafe_vectorized_intrinsics_2x4packed_bump')}
    assert {tuple(name.split('::')[:2]) for name in core['added']} == {
        ('Lokad.Onnx.MathOps', 'PackedColumnTailEightRows'),
        ('Lokad.Onnx.MathOps', 'PackedColumnMaskedEightRows')}
    assert compatible['selected'] == models['identities']['selected']
    assert compatible['candidate'] == models['identities']['candidate']
    assert compatible['selected']['Lokad.Onnx.dll']['sha256'] == '47984318b082710c3a4f57a85b1500d49d7c1236c04b1234477d19e48d11207c'
    assert compatible['candidate']['Lokad.Onnx.dll']['sha256'] == '7cac67880fa9a4d519ac18e5887f47f48f0f14903bdf74cc6561b45c851e4f27'
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
    assert not compatible['component_screen_admitted']
    assert compatible['runtime_diagnosis']['sha256'] == '75f3b25985a8ff4efdc7465de0444b3e667deb0c5922755b0da8a8a5770601c4'
    assert compatible['arithmetic_qualification']['sha256'] == '558f2a523febd0d794bc7da6cdae3381513b7ff3d75dfd4ca4f133f618821cc8'
    assert len(compatible['failed_component_controls']) == 38
    assert all(not row['passed'] for row in compatible['failed_component_controls'])
    assert not models.get('release_admitted', False)
