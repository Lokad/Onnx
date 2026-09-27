"""Bind original Replay through actual root to the single rational arithmetic edit."""


def validate(compatible, models, previous, app, consumer):
    assert compatible['passed'] and compatible['original_public_bindings_preserved']
    assert compatible['all_original_method_flags_preserved'] and compatible['underlying_methods_reconciled'] == 3979
    assert compatible['all_data_methods_exact'] and compatible['no_consumer_or_product_build']
    assert compatible['changed_core_methods'] == ['Sigmoid']
    assert compatible['added_private_methods'] == ['SigmoidRationalVector']
    assert compatible['selected'] == models['identities']['selected']
    assert compatible['candidate'] == models['identities']['candidate']
    assert compatible['qualified_model_product'] == previous['identities']['candidate']
    assert models['passed'] and previous['passed'] and previous['reference_provenance_verified']
    assert previous['consumer'] == consumer
    assert consumer['sha256'] == 'a50d3e965cf480844559b1f5856e3de318b5c8a269742a9e6504afb9cee6c8c0'
    assert app['passed'] and app['performance']['admitted']
    assert app['identities'] == dict(current=models['identities']['selected'], candidate=models['identities']['candidate'])
    for key, count in [('controls', 63), ('gates', 21)]:
        rows = app['performance'][key]
        assert len(rows) == count and all(row['passed'] for row in rows)
    assert not compatible['component_screen_admitted']
    assert len(compatible['failed_component_controls']) == 13
    assert all(not r['passed'] for r in compatible['failed_component_controls'])
    assert [r['name'] for r in compatible['failed_component_cases']] == ['scalar-option', 'double', 'empty', 'scalar']
    assert all(not r['passed'] for r in compatible['failed_component_cases'])
    assert not models.get('release_admitted', False)
