"""Verify the VM closure and run the same local audit against recovery receipts."""
import inspect
import audit


base=audit.BASE;collected=base/'collected'
assert audit.pin(collected/'remote-closed.json')['sha256']=='50e487a962ed0916d15344bc6bfc21028c505fe68ecbcc4c5d94ca0ead860f46'
proof=audit.read(collected/'remote-closed.json');assert proof['passed']
for name,wanted in proof['files'].items():
    path=collected/name if (collected/name).is_file() else base/'payload'/name
    if not path.is_file():path=base/'closure-supplement'/name
    assert audit.pin(path)==wanted,name
source=inspect.getsource(audit.main)
source=source.replace('results.tar.gz','results-recovered.tar.gz')
source=source.replace('collection.json','collection-recovered.json')
source=source.replace('collection-transfer.json','collection-transfer-recovered.json')
scope=dict(vars(audit));exec(compile(source,'recovered-lstm-platform-audit','exec'),scope)
scope['main']()
