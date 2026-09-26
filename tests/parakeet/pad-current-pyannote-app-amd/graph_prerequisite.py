"""Require all eight fresh graph cases for the exact current/candidate products."""
from protocol import pin,read

CASES=['e5-8tok','e5-30tok','e5-30pad128','e5-128tok','e5-512tok','dinov3','resnet50','gpt2']


def validate(value,proof,products):
    assert proof['passed'] and proof['admitted'] and proof['all_controls_passed']
    assert value['passed'] and not value['root_product_changed']
    assert value['products']==products
    assert value['clocks']==73512 and value['measured']==8640 and len(value['setups'])==72
    for name,digest in [
        ('consumer','d827e3b9f1e5158e10bd24a5ca009fa7950fd08d08bb5260b84e36b5a853ac02'),
        ('e5_consumer','0b228b2d22eef7080a64fb5fbababc24e24a94b18deae2fead5075fbf90b4930'),
        ('short_consumer','e437850d39cafd15fc4c29b45ac70a0062f1039260c52ae89ffde490e2fad354')]:
        row=value[name]
        assert row['passed'] and row['branches_locals_exceptions_equal'] and row['implementation_flags_equal']
        assert not row['product_changed'] and (row['methods'],row['unchanged_methods'])==(66,65)
        assert row['consumer']['sha256']==digest
    assert [r['key'] for r in value['performance']]==CASES
    for row in value['performance']:
        assert row['qualified'] and row['regression_passed'] and row['candidate_over_current']<=1.05
        assert [c['role'] for c in row['controls']]==['current','candidate','ort']
        assert all(c['passed'] and c['ratio']<=1.10 for c in row['controls'])
    return True


def verify_bundle(base,spec):
    folder=base/'evidence/graph-qualification';wanted=spec['graph_qualification']
    assert pin(folder/'closed.json')==wanted['closed'] and pin(folder/'analysis.json')==wanted['analysis']
    proof,value=read(folder/'closed.json'),read(folder/'analysis.json')
    assert proof['files']['analysis.json']==wanted['analysis']
    products={role:{'Lokad.Onnx.dll':spec['identities'][label]['Lokad.Onnx.dll']}
              for role,label in [('current','selected'),('candidate','candidate')]}
    validate(value,proof,products)
    return dict(passed=True,closed=wanted['closed'],cases=CASES,clocks=73512,measured=8640)
