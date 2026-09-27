"""Require exact compiled ancestry and full application admission for this fixed pair."""


def validate(compatible, models, previous, app, consumer):
    assert compatible['passed'] and compatible['original_public_bindings_preserved']
    assert compatible['all_original_method_flags_preserved'] and compatible['underlying_methods_reconciled'] == 3980
    assert compatible['all_data_methods_exact'] and compatible['no_consumer_or_product_build']
    compiled = compatible['compiled']
    assert compiled['passed'] and compiled['public_surface_equal'] and compiled['flags_equal'] and compiled['assembly_metadata_equal']
    core, data = compiled['assemblies']
    assert core['unchanged'] == 3281 and data['unchanged'] == 697
    assert len(core['changed']) == 2 and len(core['added']) == 1 and not data['changed'] and not data['added']
    assert [s.split('::')[1] for s in core['changed']] == ['ResolvePackedKernel', 'RunPreparedPackedRows']
    assert core['added'][0].startswith('Lokad.Onnx.PreparedSingleRowKernel::Multiply::')
    assert compatible['selected'] == models['identities']['selected']
    assert compatible['candidate'] == models['identities']['candidate']
    assert compatible['selected']['Lokad.Onnx.dll']['sha256'] == '65f15a41764660af6166c9a965943b6f28cac0c9505117d0fd4beaa2687f9d03'
    assert compatible['candidate']['Lokad.Onnx.dll']['sha256'] == 'af19b3b4429a07f7966b5e35ee04e8a31316f45991c300f3683a591caf5e9374'
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
    assert not compatible['component_screen_admitted'] and compatible['failed_first_call_prediction']
    assert len(compatible['failed_component_controls']) == len(compatible['failed_component_cases']) == 2
    assert all(not r['passed'] for r in compatible['failed_component_controls']+compatible['failed_component_cases'])
    assert not models.get('release_admitted', False)
