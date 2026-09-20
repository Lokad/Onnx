"""Check that storage metadata corruption cannot pass the additional gate."""
import copy,unittest
from sharing_protocol import validate_sharing


def fixture():
    graph=dict(initializers=[dict(name='w',tensor_name='w',type='Float',shape=[1024],bytes=4096,sha256='1'*64)],
        nodes=[dict(Name='n',op='MatMul',Inputs=['x','w'],Outputs=['y'])],packed_bytes=0)
    before=dict(logical_shared_bytes=635187200,shared_arrays=88,shared_payload_bytes=635187200,unique_arrays=499,
        unique_payload_bytes=1349819052,first=copy.deepcopy(graph),past=copy.deepcopy(graph))
    return dict(weight_sharing=dict(before=before,after=copy.deepcopy(before)))


class ProtocolTests(unittest.TestCase):
    def test_accepts(self):
        self.assertEqual(635187200,validate_sharing(fixture())['logical_shared_bytes'])

    def test_detects_after_mutation(self):
        for edit in [lambda p:p.update(logical_shared_bytes=1),lambda p:p.update(unique_arrays=500),
            lambda p:p['past']['initializers'][0].update(sha256='0'*64),lambda p:p['first']['nodes'][0].update(op='Gemm')]:
            with self.subTest(edit=edit):
                v=fixture();edit(v['weight_sharing']['after'])
                with self.assertRaises(AssertionError):validate_sharing(v)

    def test_detects_identical_corruption(self):
        for edit in [lambda p:p.update(logical_shared_bytes=635187199),lambda p:p.update(shared_payload_bytes=635187199),
            lambda p:p.update(unique_arrays=1),lambda p:p.update(unique_payload_bytes=1),lambda p:p.update(shared_arrays=True),
            lambda p:p['past']['initializers'][0].update(sha256='invalid'),lambda p:p['first'].update(initializers=[]),
            lambda p:p['first']['initializers'][0].update(shape=[-1]),lambda p:p['past']['nodes'][0].update(Inputs=None)]:
            with self.subTest(edit=edit):
                v=fixture();edit(v['weight_sharing']['before']);v['weight_sharing']['after']=copy.deepcopy(v['weight_sharing']['before'])
                with self.assertRaises(AssertionError):validate_sharing(v)


if __name__=='__main__':unittest.main()
