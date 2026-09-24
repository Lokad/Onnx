"""Require a clean consumer build and the exact unchanged product before timing."""
import base64
import json
from checks import resources
from run import BASE, PRELUDE, pin, read, write, prepared, ssh


def main():
    prepared();assert not (BASE/'build-review.json').exists()
    rows=resources('build');folder=BASE/'build-collected'
    assert [r['name'] for r in rows]==['sdk-version','restore','build']
    built=read(folder/'built.json');spec=read(BASE/'bundle/spec.json')
    assert built['core']==spec['core']==pin(folder/'runtime/Lokad.Onnx.dll')
    for name,wanted in built['runtime'].items():assert pin(folder/name)==wanted,name
    assert built['consumer']==pin(folder/'runtime/CopyCost.dll')
    assert built['manifest']==pin(folder/'manifest.json')
    manifest=read(folder/'manifest.json')
    assert manifest==dict(read(BASE/'bundle/cases.json'),consumer_sha256=built['consumer']['sha256'])
    log=(folder/'logs/build.stdout').read_text(encoding='utf8')
    assert '0 Warning(s)' in log and '0 Error(s)' in log
    assert not (folder/'logs/build.stderr').read_text().strip()
    review=dict(passed=True,core_unchanged=True,no_product_build=True,built=pin(folder/'built.json'),
        spec=pin(BASE/'bundle/spec.json'),collection=pin(folder/'build-collection.json'),resources=rows,reviewer=pin(__file__))
    write(BASE/'build-review.json',review)
    encoded=base64.b64encode((BASE/'build-review.json').read_bytes()).decode()
    result=ssh(PRELUDE+f'''
import base64
from remote import read,pin,verify,live,idle
verify();idle();state=read(base/'build-state.json')
assert state['complete'] and state['code']==0 and not live(state['supervisor'])
assert all(not live(dict(pid=int(p),birth=b)) for r in state['runs'] for p,b in r['members'].items())
assert pin(base/'built.json')=={review['built']!r}
with (base/'build-review.json').open('xb') as stream:stream.write(base64.b64decode({encoded!r}))
print(json.dumps(dict(passed=True,review=pin(base/'build-review.json'))))
''')
    assert result['review']==pin(BASE/'build-review.json')
    write(BASE/'build-review-transferred.json',result)
    print(json.dumps(dict(passed=True,review=result['review'],consumer=built['consumer'],resources=rows)))


if __name__=='__main__':main()
