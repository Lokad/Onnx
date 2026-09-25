"""Bind reused consumers to their original compiled proof and actual products."""
from protocol import pin, read
from checks import inventory


def verify_reuse(base, spec):
    sources = {
        'selected': ('current', 'e36fe9c83405608659ebdc4d673c185c8805a5414a6b01d401904209d59417dd'),
        'candidate': ('reuse', '8471ed0c4f9cdca7ba32edf2ba7ef744bedae4c0b0b037af52f1c87642230595'),
    }
    for role, (label, digest) in sources.items():
        path = base/'evidence'/(label+'-closed.json')
        proof = read(path)
        assert proof['passed'] and pin(path)['sha256'] == digest
        for suffix in ['dll', 'deps.json', 'runtimeconfig.json']:
            name = 'GraphQualification.'+suffix
            assert pin(base/'runtimes'/role/name) == proof['files']['collected/built/'+name]
        assert pin(base/'runtimes'/role/'GraphQualification.dll') == spec['consumers'][role]
        for name, wanted in spec['identities'][role].items():
            assert pin(base/'runtimes'/role/name) == wanted
    proof = read(base/'evidence/reuse-closed.json')
    for name, original in [('reuse-built.json', 'built.json'),
                           ('reuse-instructions.json', 'consumer-inventory/instructions.json'),
                           ('reuse-review.json', 'consumer-inventory/review.json')]:
        assert pin(base/'evidence'/name) == proof['files']['collected/'+original]
    built = read(base/'evidence/reuse-built.json')
    assert built['passed'] and built['consumer'] == spec['consumers']['candidate']
    assert spec['identities']['selected']['Lokad.Onnx.Data.dll']['sha256'] == spec['old_data']
    assert spec['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256'] == spec['new_data']
    result = inventory(read(base/'evidence/reuse-instructions.json'), spec, built)
    assert result == read(base/'evidence/reuse-review.json')
    return dict(**result, consumer_rebuilt=False, retained_closure=sources['candidate'][1])
