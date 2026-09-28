"""Require the exact admitted collapsed-axis transpose pair and preserved graph-consumer bindings."""


def validate(reports, compatible, scope):
    assert set(reports) == {'graph', 'models', 'app', 'shared', 'pyannote'}
    assert all(r['passed'] for r in reports.values())
    pair = reports['models']['identities']
    assert pair['selected']['Lokad.Onnx.dll']['sha256'] == '4e97e2ae97e534b99281a361790ac35865b3d8c6171293f36109b24a354eec57'
    assert pair['candidate']['Lokad.Onnx.dll']['sha256'] == 'c471f5d1ead5889b00b141ce34f2a7689cfe179fbf513da4ce5db84ed01ce277'
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
    assert compatible['all_original_method_flags_preserved'] and compatible['underlying_methods_reconciled'] == 3985
    core, data = compatible['compiled_scope']
    assert core['assembly'] == 'Lokad.Onnx.dll' and core['unchanged'] == 3287
    assert core['original'] == 3288 and len(core['changed']) == 1
    assert {tuple(name.split('::')[:2]) for name in core['changed']} == {
        ('Lokad.Onnx.Tensor`1[T]', 'TransposeInto')}
    assert data == dict(assembly='Lokad.Onnx.Data.dll', original=697, unchanged=697, changed=[])
    assert compatible['all_data_methods_exact'] and compatible['no_consumer_or_product_build']
    assert compatible['selected'] == pair['selected'] and compatible['candidate'] == pair['candidate']
    assert compatible['qualified_model_product']['Lokad.Onnx.dll'] == graph['products']['candidate']['Lokad.Onnx.dll']
    assert compatible['focused_contracts']['sha256'] == 'c0c063bb3f73eed2d8d7372d17dd1da3ce4f6a962e190ef101b504fb245e4ecd'
    assert compatible['inputs']['artifacts/parakeet-attention-owned-root-recovery-amd-20260928/closed.json']['sha256'] == 'ee4a38ff671cc3fd8cd608c0dc3008f5c1b99f61ba6c139f3deeaf0e9039305e'
    for key, digest in [
        ('consumer', 'd827e3b9f1e5158e10bd24a5ca009fa7950fd08d08bb5260b84e36b5a853ac02'),
        ('e5_consumer', '0b228b2d22eef7080a64fb5fbababc24e24a94b18deae2fead5075fbf90b4930'),
        ('short_consumer', 'e437850d39cafd15fc4c29b45ac70a0062f1039260c52ae89ffde490e2fad354')]:
        assert graph[key]['passed'] and graph[key]['implementation_flags_equal']
        assert graph[key]['consumer']['sha256'] == digest
    assert scope['passed'] and scope['inference_calls'] == 0 and not scope['runtime_dispatch_measured']
    assert len(scope['models']) == 4 and sum(len(m['cases']) for m in scope['models']) == 8
    # Reuse the closed export/case census only. Absence of Sigmoid is irrelevant
    # to this transpose-dispatch change and does not establish runtime dispatch.
    assert {case for m in scope['models'] for case in m['cases']} == {
        'e5-8tok', 'e5-30tok', 'e5-30pad128', 'e5-128tok', 'e5-512tok', 'dinov3', 'resnet50', 'gpt2'}
    return {role: {'Lokad.Onnx.dll': pair[label]['Lokad.Onnx.dll']} for role, label in [('current', 'selected'), ('candidate', 'candidate')]}
