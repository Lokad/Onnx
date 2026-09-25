"""Exercise accounting failures that could otherwise invent a performance lead."""
import copy
import unittest

from analyze import controls, feed_forward_cost, node_cost


def sample(prepared=False, scaled=True):
    k, n = (4096, 1024) if scaled else (1024, 4096)
    stages = [dict(stage=8, duration_ticks=10), dict(stage=1005, duration_ticks=10),
              dict(stage=1005, duration_ticks=10)]
    stages += [dict(stage=1006, duration_ticks=20)] if prepared else [
        dict(stage=code, duration_ticks=10) for code in range(1000, 1005)]
    node = dict(id=4, op='MatMul', detail=f'A:float:1x106x{k},onnx::W:float:{k}x{n}', stages=stages)
    wall = dict(NodeId=4, StartTicks=100, EndTicks=300)
    route = dict(request=0, clip='clip', pass_index=0, name='projection', native_name='native',
        m=106, frames=106, k=k, n=n, alpha=.5 if scaled else 1.,
        route='mapped-allowed' if prepared else 'unmapped', scratch_bytes=0 if prepared else 16777216,
        copy_bytes=0, ort_seconds=.001)
    group = dict(ort_name='native', managed_nodes=['projection', 'scale'] if scaled else ['projection'],
                 shape=[k, n], alpha=route['alpha'])
    scale = dict(id=5, op='Mul', detail='', stages=[dict(stage=8, duration_ticks=10)])
    scale_wall = dict(NodeId=5, StartTicks=301, EndTicks=351)
    nodes = {'scale': dict(node=scale, cost=node_cost(scale, scale_wall, 10_000_000))}
    return node, wall, route, group, nodes


def roles():
    result = {}
    for mode in ['clock', 'stages', 'markers']:
        requests = [dict(name=f'clip-{c}', pass_index=p, seconds=1.) for p in [1, 2, 3] for c in range(20)]
        result[mode] = dict(corpus_seconds=20., requests=requests)
    return result


class CostAccounting(unittest.TestCase):
    def test_keeps_residual_and_both_output_preparation_intervals(self):
        node, wall, *_ = sample()
        value = node_cost(node, wall, 10_000_000)
        self.assertEqual(value['stage_counts'][1005], 2)
        self.assertAlmostEqual(value['wall_seconds'], sum(value['stage_seconds'].values()) + value['residual_seconds'])
        self.assertGreater(value['residual_seconds'], 0)

    def test_different_clock_units_are_reconciled_exactly(self):
        node, wall, *_ = sample()
        wall['EndTicks'] = 20_100
        value = node_cost(node, wall, 1_000_000_000)
        self.assertAlmostEqual(value['wall_seconds'], .000020)
        self.assertAlmostEqual(value['residual_seconds'], .000012)

    def test_negative_residual_is_rejected_not_clamped(self):
        node, wall, *_ = sample()
        wall['EndTicks'] = 179
        with self.assertRaises(AssertionError):
            node_cost(node, wall, 10_000_000)

    def test_negative_noninteger_and_unknown_stage_values_are_rejected(self):
        for field, value in [('duration_ticks', -1), ('duration_ticks', 1.5), ('duration_ticks', True), ('stage', 999)]:
            with self.subTest(field=field, value=value):
                node, wall, *_ = sample()
                node['stages'][0][field] = value
                with self.assertRaises(AssertionError):
                    node_cost(node, wall, 10_000_000)

    def test_node_identity_must_match_enclosing_clock(self):
        node, wall, *_ = sample()
        wall['NodeId'] += 1
        with self.assertRaises(AssertionError):
            node_cost(node, wall, 10_000_000)

    def test_scaled_group_includes_the_entire_separate_multiply(self):
        value = feed_forward_cost(*sample(), 10_000_000, 'markers')
        self.assertAlmostEqual(value['complete_group_seconds'], .000025)
        self.assertAlmostEqual(value['scale_wall_seconds'], .000005)

    def test_missing_fused_scale_is_rejected(self):
        node, wall, route, group, nodes = sample()
        group['managed_nodes'] = ['projection']
        with self.assertRaises(AssertionError):
            feed_forward_cost(node, wall, route, group, nodes, 10_000_000, 'markers')

    def test_wrong_runtime_shape_is_rejected(self):
        node, wall, route, group, nodes = sample()
        node['detail'] = node['detail'].replace('1x106', '1x107')
        with self.assertRaises(AssertionError):
            feed_forward_cost(node, wall, route, group, nodes, 10_000_000, 'markers')

    def test_unexplained_prepared_or_dynamic_route_is_rejected(self):
        node, wall, route, group, nodes = sample()
        route['route'] = 'mapped-allowed'
        with self.assertRaises(AssertionError):
            feed_forward_cost(node, wall, route, group, nodes, 10_000_000, 'markers')

    def test_prepared_route_has_no_runtime_packing_stages(self):
        value = feed_forward_cost(*sample(prepared=True, scaled=False), 10_000_000, 'markers')
        self.assertEqual(value['stage_counts'][1006], 1)
        self.assertNotIn(1001, value['stage_counts'])

    def test_controls_retain_failed_observer_effect(self):
        values = roles()
        values['markers']['corpus_seconds'] *= 1.06
        result = controls(values)
        self.assertEqual(len(result['repeatability']), 63)
        self.assertEqual(len(result['observer_effects']), 2)
        self.assertFalse(result['usable_for_candidate_selection'])
        self.assertTrue(all(r['passed'] for r in result['repeatability']))

    def test_large_apparent_speedup_also_invalidates_observer(self):
        values = roles()
        values['stages']['corpus_seconds'] *= .94
        self.assertFalse(controls(values)['usable_for_candidate_selection'])

    def test_no_repeatability_sample_is_discarded(self):
        values = roles()
        values['markers']['requests'][0]['seconds'] = 2.
        result = controls(values)
        self.assertFalse(result['usable_for_candidate_selection'])
        failed = next(r for r in result['repeatability'] if r['name'] == 'markers:clip-0:repeatability')
        self.assertEqual(failed['values'], [2., 1., 1.])

    def test_duplicate_pass_cannot_replace_a_missing_observation(self):
        values = roles()
        values['markers']['requests'][20] = copy.deepcopy(values['markers']['requests'][0])
        with self.assertRaises(AssertionError):
            controls(values)


if __name__ == '__main__':
    unittest.main()
