"""Reject missing wall nodes, reordered state and truncated decoder traces."""
import copy
import unittest
import numpy as np
from audit import clocks, trajectory


class AuditTests(unittest.TestCase):
    def test_wall_clocks_require_all_nodes_in_order_and_within_graph(self):
        graph = [dict(id=1, op='MatMul'), dict(id=2, op='Add')]
        row = dict(frequency=10, reset_start_ticks=0, reset_end_ticks=1, start_ticks=2, end_ticks=10, pass_=1,
                   nodes=[dict(id=1, op='MatMul', start_ticks=3, end_ticks=5), dict(id=2, op='Add', start_ticks=6, end_ticks=9)])
        row['pass'] = row.pop('pass_')
        self.assertEqual(float(clocks(row, graph)['MatMul']), .2)
        for mutation in ('missing', 'overlap', 'wrong-op', 'outside', 'duplicate'):
            damaged = copy.deepcopy(row)
            if mutation == 'missing': damaged['nodes'].pop()
            elif mutation == 'overlap': damaged['nodes'][1]['start_ticks'] = 4
            elif mutation == 'wrong-op': damaged['nodes'][0]['op'] = 'Div'
            elif mutation == 'outside': damaged['nodes'][1]['end_ticks'] = 11
            else: damaged['nodes'][1]['id'] = 1
            with self.assertRaises(AssertionError): clocks(damaged, graph)
        row['pass'] = 0
        with self.assertRaises(AssertionError): clocks(row, graph)
        row['nodes'] = []; self.assertEqual(clocks(row, graph), {})

    def fixture(self):
        pcm = np.ones(2560, dtype='<f4'); features = np.ones((1, 128, 17), dtype='<f4')
        hidden = np.arange(3072, dtype='<f4').reshape(1, 1024, 3); zero = np.zeros((2, 1, 640), dtype='<f4')
        front = dict(inputs=dict(waveforms=pcm.reshape(1, -1), waveforms_lens=np.array([2560], dtype='<i8')),
                     outputs=dict(features=features, features_lens=np.array([17], dtype='<i8')))
        encoder = dict(inputs=dict(audio_signal=features, length=np.array([17], dtype='<i8')),
                       outputs=dict(outputs=hidden, encoded_lengths=np.array([3], dtype='<i8')))
        calls = [front, encoder]
        for frame, target, token, duration, state, returned in [(0, 8192, 3, 0, 0, 1), (0, 3, 8192, 1, 1, 9), (1, 3, 4, 2, 1, 3)]:
            logits = np.full((1, 1, 1, 8198), -2, dtype='<f4'); logits.flat[token] = 3; logits.flat[8193+duration] = 3
            calls.append(dict(inputs=dict(encoder_outputs=hidden[:, :, frame:frame+1].copy(), targets=np.array([[target]], dtype='<i4'),
                target_length=np.array([1], dtype='<i4'), input_states_1=zero+state, input_states_2=zero+2*state),
                outputs=dict(outputs=logits, prednet_lengths=np.array([1], dtype='<i4'), output_states_1=zero+returned, output_states_2=zero+2*returned)))
        pieces = ['x']*8193; pieces[3] = '\u2581first'; pieces[4] = '\u2581second'; pieces[-1] = '<blk>'
        expected = dict(text='first second', token_ids=[3, 4], frame_indices=[0, 1], duration_frames=[0, 2],
                        stop_reason='EndOfAudio', encoded_frames=3, decoder_calls=3)
        return calls, expected, pcm, pieces

    def test_own_states_blank_retention_frames_and_complete_trajectory(self):
        calls, expected, pcm, pieces = self.fixture()
        self.assertEqual(trajectory(calls, expected, pcm, pieces), expected)
        for mutation in ('blank-state', 'frame', 'target', 'truncated', 'length', 'different-logit-decision'):
            damaged = copy.deepcopy(calls)
            if mutation == 'blank-state': damaged[4]['inputs']['input_states_1'][:] = 9
            elif mutation == 'frame': damaged[4]['inputs']['encoder_outputs'][:] = 0
            elif mutation == 'target': damaged[3]['inputs']['targets'][:] = 8192
            elif mutation == 'truncated': damaged.pop()
            elif mutation == 'length': damaged[3]['outputs']['prednet_lengths'][:] = 2
            else: damaged[2]['outputs']['outputs'].flat[6] = 9
            with self.assertRaises(AssertionError): trajectory(damaged, expected, pcm, pieces)


if __name__ == '__main__': unittest.main()
