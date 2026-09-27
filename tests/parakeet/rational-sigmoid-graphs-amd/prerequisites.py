"""Require the exact admitted arithmetic pair and preserved graph-consumer bindings."""


def validate(reports, compatible, scope):
    assert set(reports) == {'graph', 'models', 'app', 'shared', 'pyannote'}
    assert all(r['passed'] for r in reports.values())
    pair = reports['models']['identities']
    assert pair['selected']['Lokad.Onnx.dll']['sha256'] == '8bb22038d0b4c09b56b2cdae06c49c165b8e646bc73ca28ad400f4ace0bfc659'
    assert pair['candidate']['Lokad.Onnx.dll']['sha256'] == '946ddfb66492c48a0fc6078ecbe1957ac494ff5d9ff0be42259d70e66e3b1f24'
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
    assert compatible['all_original_method_flags_preserved'] and compatible['underlying_methods_reconciled'] == 3979
    assert compatible['changed_core_methods'] == ['Sigmoid'] and compatible['added_private_methods'] == ['SigmoidRationalVector']
    assert compatible['selected'] == pair['selected'] and compatible['candidate'] == pair['candidate']
    assert compatible['qualified_model_product']['Lokad.Onnx.dll'] == graph['products']['candidate']['Lokad.Onnx.dll']
    assert not compatible['component_screen_admitted']
    assert len(compatible['failed_component_controls']) == 13 and len(compatible['failed_component_cases']) == 4
    assert all(not r['passed'] for r in compatible['failed_component_controls']+compatible['failed_component_cases'])
    for key, digest in [
        ('consumer', 'd827e3b9f1e5158e10bd24a5ca009fa7950fd08d08bb5260b84e36b5a853ac02'),
        ('e5_consumer', '0b228b2d22eef7080a64fb5fbababc24e24a94b18deae2fead5075fbf90b4930'),
        ('short_consumer', 'e437850d39cafd15fc4c29b45ac70a0062f1039260c52ae89ffde490e2fad354')]:
        assert graph[key]['passed'] and graph[key]['implementation_flags_equal']
        assert graph[key]['consumer']['sha256'] == digest
    assert scope['passed'] and scope['inference_calls'] == 0 and not scope['runtime_dispatch_measured']
    assert len(scope['models']) == 4 and sum(len(m['cases']) for m in scope['models']) == 8
    assert all(m['sigmoid_nodes'] == m['functions'] == 0 for m in scope['models'])
    return {role: {'Lokad.Onnx.dll': pair[label]['Lokad.Onnx.dll']} for role, label in [('current', 'selected'), ('candidate', 'candidate')]}
