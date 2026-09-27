"""Require the exact admitted prepared-row pair and preserved graph-consumer bindings."""


def validate(reports, compatible, scope):
    assert set(reports) == {'graph', 'models', 'app', 'shared', 'pyannote'}
    assert all(r['passed'] for r in reports.values())
    pair = reports['models']['identities']
    assert pair['selected']['Lokad.Onnx.dll']['sha256'] == '65f15a41764660af6166c9a965943b6f28cac0c9505117d0fd4beaa2687f9d03'
    assert pair['candidate']['Lokad.Onnx.dll']['sha256'] == 'af19b3b4429a07f7966b5e35ee04e8a31316f45991c300f3683a591caf5e9374'
    assert all(reports[k]['identities'] == pair for k in ['shared', 'pyannote'])
    app = reports['app']; assert app['identities'] == dict(current=pair['selected'], candidate=pair['candidate'])
    assert app['performance']['admitted']
    for key, count in [('controls', 63), ('gates', 21)]:
        assert len(app['performance'][key]) == count and all(r['passed'] for r in app['performance'][key])
    pyannote = reports['pyannote']
    assert pyannote['identity_guards']['passed'] and pyannote['identity_guards']['probes'] == 4
    assert pyannote['results']['candidate']['complete_public_results_exact']
    graph = reports['graph']
    assert len(graph['performance']) == 8 and all(r['qualified'] for r in graph['performance'])
    assert compatible['passed'] and compatible['original_public_bindings_preserved']
    assert compatible['all_original_method_flags_preserved'] and compatible['underlying_methods_reconciled'] == 3980
    compiled = compatible['compiled']
    assert compiled['passed'] and compiled['public_surface_equal'] and compiled['flags_equal'] and compiled['assembly_metadata_equal']
    core, data = compiled['assemblies']
    assert core['assembly'] == 'Lokad.Onnx.dll' and core['unchanged'] == 3281
    assert [name.split('::')[1] for name in core['changed']] == ['ResolvePackedKernel', 'RunPreparedPackedRows']
    assert len(core['added']) == 1 and core['added'][0].startswith('Lokad.Onnx.PreparedSingleRowKernel::Multiply::')
    assert data == dict(assembly='Lokad.Onnx.Data.dll', unchanged=697, changed=[], added=[])
    assert compatible['all_data_methods_exact'] and compatible['no_consumer_or_product_build']
    assert compatible['failed_first_call_prediction']
    assert compatible['selected'] == pair['selected'] and compatible['candidate'] == pair['candidate']
    assert compatible['qualified_model_product']['Lokad.Onnx.dll'] == graph['products']['candidate']['Lokad.Onnx.dll']
    assert not compatible['component_screen_admitted']
    assert len(compatible['failed_component_controls']) == len(compatible['failed_component_cases']) == 2
    assert all(not r['passed'] for r in compatible['failed_component_controls']+compatible['failed_component_cases'])
    for key, digest in [
        ('consumer', 'd827e3b9f1e5158e10bd24a5ca009fa7950fd08d08bb5260b84e36b5a853ac02'),
        ('e5_consumer', '0b228b2d22eef7080a64fb5fbababc24e24a94b18deae2fead5075fbf90b4930'),
        ('short_consumer', 'e437850d39cafd15fc4c29b45ac70a0062f1039260c52ae89ffde490e2fad354')]:
        assert graph[key]['passed'] and graph[key]['implementation_flags_equal']
        assert graph[key]['consumer']['sha256'] == digest
    assert scope['passed'] and scope['inference_calls'] == 0 and not scope['runtime_dispatch_measured']
    assert len(scope['models']) == 4 and sum(len(m['cases']) for m in scope['models']) == 8
    # Reuse the closed export/case census only. Absence of Sigmoid is irrelevant
    # to this addressing-only change and does not establish runtime dispatch.
    assert {case for m in scope['models'] for case in m['cases']} == {
        'e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok', 'dinov3', 'resnet50', 'gpt2'}
    return {role: {'Lokad.Onnx.dll': pair[label]['Lokad.Onnx.dll']} for role, label in [('current', 'selected'), ('candidate', 'candidate')]}
