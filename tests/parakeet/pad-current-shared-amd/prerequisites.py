"""Bind the unchanged Replay consumer through the reviewed root transition."""


def validate(compatible, models, previous, app, consumer):
    assert compatible['passed'] and compatible['original_public_bindings_preserved']
    assert compatible['all_original_method_flags_preserved'] and compatible['underlying_methods_reconciled']==3978
    assert compatible['padding']['only_public_pad_changed'] and compatible['padding']['original_padcore_exact']
    assert compatible['selected']==models['identities']['selected']
    assert compatible['candidate']==models['identities']['candidate']
    assert compatible['qualified_model_product']==previous['identities']['candidate']
    assert previous['passed'] and previous['reference_provenance_verified']
    assert previous['consumer']==consumer
    assert consumer['sha256']=='a50d3e965cf480844559b1f5856e3de318b5c8a269742a9e6504afb9cee6c8c0'
    assert app['passed'] and app['performance']['admitted']
    assert app['identities']==dict(current=models['identities']['selected'],candidate=models['identities']['candidate'])
    assert not compatible['component_screen_admitted'] and len(compatible['failed_component_controls'])==6
    assert not models.get('release_admitted',False)
