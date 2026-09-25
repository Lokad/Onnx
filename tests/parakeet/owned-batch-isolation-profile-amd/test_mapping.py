import copy
import unittest
from diagnose import descriptors_equal


class MappingChecks(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(graph='encoder',id=1,name='projection',op='MatMul',inputs=['a','b'],
            outputs=['c'],calls=60,constant_inputs=[None,dict(dims=[8,16])],ticks=50,corpus_seconds=.5)]

    def test_only_clocks_may_differ(self):
        after = copy.deepcopy(self.rows); after[0].update(ticks=1,corpus_seconds=.01)
        descriptors_equal(self.rows,after)

    def test_graph_identity_shape_edges_and_calls_are_required(self):
        for key,value in [('graph','decoder'),('id',2),('op','Mul'),('inputs',['b','a']),
            ('outputs',['d']),('calls',59),('constant_inputs',[None,dict(dims=[4,32])])]:
            with self.subTest(key=key), self.assertRaises(AssertionError):
                after = copy.deepcopy(self.rows); after[0][key] = value
                descriptors_equal(self.rows,after)

    def test_duplicate_or_missing_nodes_are_rejected(self):
        for after in [[],self.rows*2]:
            with self.assertRaises(AssertionError): descriptors_equal(self.rows,after)


if __name__ == '__main__': unittest.main()
