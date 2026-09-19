import copy
import math
import unittest
import numpy as np
from audit import input_hash, scaled_error, validate_model


def fixture():
    inputs = dict(input_ids=[0, 2], attention_mask=[1, 1], token_type_ids=[0, 0])
    identities = dict(model_sha256='model', tokenizer_sha256='tokenizer', input_sha256=input_hash(inputs))
    meta = dict(protocol='zero-block-product-descriptive-v1', core_sha256='core', probe_sha256='probe',
                oracles={'e5-8tok': dict(sha256='fixture', manifest=identities)})
    bundle = dict(files={'model/bin/Lokad.Onnx.Campaign.dll': dict(sha256='runner')})
    value = dict(schema=3, protocol=meta['protocol'], diagnostic_only=True, name='e5-8tok', mode='memory',
                 role='controlA', optimization='Memory', native=None, parallelism=1, settings={}, affinity=4,
                 runtime='.NET 10.0.8', avx2=True, avx512=True, vector_width=8, core_sha256='core', probe_sha256='probe',
                 runner_sha256='runner', fixture_sha256='fixture', inputs=inputs, inputs_unchanged=True,
                 held_outputs_unchanged=True, frequency=1_000_000_000, load_ticks=1, first_execute_ticks=1,
                 conditioning=[15_000_000_000, 15_000_000_000], conditioning_seconds=30., **identities)
    for boundary in ('execute', 'request'):
        value[boundary] = dict(include_reset=boundary == 'request', ticks=[1_000_000] * 33, allocated_bytes=99,
                               before=dict(gc=[1, 2, 3]), after=dict(gc=[2, 3, 3]))
    return value, meta, bundle


class AuditTests(unittest.TestCase):
    def test_valid_worker_and_full_output_difference(self):
        value, meta, bundle = fixture()
        result = validate_model(value, meta, bundle, 'e5-8tok', 'controlA', {})
        self.assertEqual([1.] * 33, result['execute']['samples_ms'])
        self.assertEqual([1, 1, 0], result['request']['gc'])
        self.assertAlmostEqual(5e-5, scaled_error(np.array([1., 2.0001]), np.array([1., 2.])))

    def test_identity_contract_and_clock_refusals(self):
        changes = dict(schema=2, protocol='old', mode='default', optimization='Speed', native={}, parallelism=2,
                       settings={'DOTNET_JitOSR': '0'}, affinity=3, runtime='.NET 10.0.12', vector_width=16,
                       core_sha256='other', probe_sha256='other', runner_sha256='other', fixture_sha256='other',
                       input_sha256='other', inputs_unchanged=False, held_outputs_unchanged=False,
                       frequency=0, load_ticks=-1, first_execute_ticks=math.nan, conditioning_seconds=29.)
        for key, bad in changes.items():
            value, meta, bundle = fixture(); value[key] = bad
            with self.assertRaises(ValueError, msg=key):
                validate_model(value, meta, bundle, 'e5-8tok', 'controlA', {})
        for conditioning in ([], [29_000_000_000], [30_000_000_000, 1], [math.nan], [True]):
            value, meta, bundle = fixture(); value['conditioning'] = conditioning
            with self.assertRaises(ValueError):
                validate_model(value, meta, bundle, 'e5-8tok', 'controlA', {})

    def test_bad_samples_allocations_and_gc_refused(self):
        changes = [('ticks', [1] * 32), ('ticks', [1] * 32 + [0]), ('ticks', [1] * 32 + [math.inf]),
                   ('allocated_bytes', -1), ('before', dict(gc=[3, 4, 5])), ('include_reset', True)]
        for key, bad in changes:
            value, meta, bundle = fixture(); value['execute'][key] = bad
            with self.assertRaises(ValueError):
                validate_model(value, meta, bundle, 'e5-8tok', 'controlA', {})

    def test_full_array_nonfinite_and_mismatch_refused(self):
        for actual in (np.array([1., math.nan]), np.array([1., math.inf]), np.array([1.]), np.array([1., 2.001])):
            with self.assertRaises(ValueError):
                scaled_error(actual, np.array([1., 2.]))
        value, _, _ = fixture(); old = input_hash(value['inputs']); value['inputs']['input_ids'][1] = 3
        self.assertNotEqual(old, input_hash(value['inputs']))


if __name__ == '__main__':
    unittest.main()
