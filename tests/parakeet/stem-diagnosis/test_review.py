import unittest
from review import spatial


class Geometry(unittest.TestCase):
    def test_stem_depthwise_one_panel_per_output_row(self):
        r=spatial(256,3,3,256,212,32,256)
        self.assertEqual((32,212,54272,162816,327680),tuple(r[k] for k in ['block_columns','blocks','matrix_calls','tensor_views','scratch_requested_bytes']))

    def test_second_depthwise_partial_panel_is_retained(self):
        r=spatial(256,3,3,256,107,16,256)
        self.assertEqual((54,16,13824),tuple(r[k] for k in ['blocks','final_matrix_n','matrix_calls']))

    def test_module_minimum_panel_exceeds_nominal_budget(self):
        r=spatial(1024,1,9,1024,1,51,1024)
        self.assertEqual((32,19,2048,1310720),tuple(r[k] for k in ['block_columns','final_matrix_n','matrix_calls','scratch_requested_bytes']))

    def test_small_problem_keeps_whole_patch(self):
        r=spatial(1,3,3,1,1,1,1)
        self.assertFalse(r['tiled']);self.assertEqual(36,r['scratch_requested_bytes'])


if __name__=='__main__':unittest.main()
