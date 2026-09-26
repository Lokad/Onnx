"""Verify the three retained compiled consumers and their complete original proofs."""
from protocol import pin,read
from checks import consumer_inventory,e5_consumer_inventory
from short_checks import consumer_inventory as short_inventory


def review(base,spec,built):
    assert built['passed'] and built['reused_consumer']
    for name,wanted in built['files'].items():assert pin(base/name)==wanted,name
    normal=consumer_inventory(read(base/'evidence/warmed-consumer/instructions.json'),spec,built)
    assert normal==read(base/'evidence/warmed-consumer/review.json')
    long=e5_consumer_inventory(read(base/'evidence/e5-consumer/instructions.json'),
        dict(previous_consumer=spec['consumer']),dict(consumer=built['e5_consumer']))
    assert long==read(base/'evidence/e5-consumer/review.json')
    short=short_inventory(read(base/'evidence/short-consumer/instructions.json'),
        dict(previous_consumer=spec['consumer']),dict(consumer=built['short_consumer']))
    assert short==read(base/'evidence/short-consumer/review.json')
    for key,folder in [('consumer','runtimes'),('e5_consumer','runtimes-e5'),('short_consumer','runtimes-short')]:
        assert built[key]==spec[key]
        for role in ['current','candidate']:
            assert pin(base/folder/role/'ReleaseBenchmark.dll')==spec[key]
            assert pin(base/folder/role/'Lokad.Onnx.dll')==spec['products'][role]['Lokad.Onnx.dll']
    return dict(normal=normal,long=long,short=short)
