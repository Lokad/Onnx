"""New campaign coverage: complete public corpus, AMD precedence and exact sums."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from candidate_protocol import ROLES, TIMING_ROLES, read
from consumer_constants import CORE, CONTROL_CORE
from operator_checks import probe
import applications
from disposable import build_caches

ROOT = Path(__file__).resolve().parents[3]
OLD = ROOT / 'artifacts/pyannote-single-panel-amd-payload-v2-20260922/payload'
sys.path.insert(0, str(OLD / 'runtime'))


class ParakeetTests(unittest.TestCase):
    def public_fixture(self, mutate, native=True):
        role = 'ort' if native else 'portable'
        label = '01-parakeet-ort' if native else '00-parakeet-managed'
        result = read(ROOT / 'artifacts/audio-amd-two-family-20260920/collected/campaign/timing' / label / 'worker/result.json')
        result.update(conformance=True, records=result['records'][:20], manifest_sha256='test', runner_sha256='test')
        manifest = read(OLD / 'manifests/production-parakeet.json')
        if not native: result.update(core_sha256=manifest['core_sha256'], data_sha256=manifest['data_sha256'])
        mutate(result)
        spec = read(OLD / 'payload.json')
        def reader(path):
            if path.name == 'result.json': return copy.deepcopy(result)
            if path.name.endswith('-parakeet.json'): return copy.deepcopy(manifest)
            if path.name == 'payload.json': return spec
            return copy.deepcopy(result['records'][int(path.stem)])
        with patch.object(applications, 'read', side_effect=reader), patch.object(applications, 'pin', return_value=dict(sha256='test')):
            return applications.inspect(Path('payload'), Path('campaign'), 'fixture', role, 'parakeet', 'conformance')

    def test_every_native_and_managed_public_case_retained(self):
        # These are retained outputs with metadata adapted only for validator tests.
        for native in [False, True]:
            self.assertEqual(len(self.public_fixture(lambda r: None, native)['records']), 20)
            changes = [lambda r: r['records'].pop(), lambda r: r.update(manifest_sha256='wrong'),
                lambda r: r.update(held_outputs_unchanged=False),
                lambda r: r['records'][0].update(ownership=False),
                lambda r: r['records'][0].update(input_sha256='wrong')]
            if native: changes.append(lambda r: r['native_settings'].update(intra_threads=2))
            for mutate in changes:
                with self.assertRaises(AssertionError): self.public_fixture(mutate, native)

    def test_missing_ort_public_result_blocks_conformance(self):
        def inspector(base, campaign, label, role, family, mode):
            if role == 'ort': raise FileNotFoundError('missing native conformance')
            return dict(records=[{}]*20)
        with patch.object(applications, 'inspect', side_effect=inspector), patch.object(applications, 'pin', return_value={}):
            with self.assertRaises(FileNotFoundError): applications.conformance(Path('payload'), Path('campaign'))

    def test_forty_amd_cases_and_positive_control_required(self):
        value = read(ROOT / 'artifacts/parakeet-single-panel-amd-consumers-v2-20260922/probe-normal.json')
        value.update(runtime='10.0.8', avx512=True)
        value['prepared_precedence'] = [dict(m=m, n=n, k=k, baseline='0'*64, candidate='0'*64, partial='1'*64,
            input_a='2'*64, input_b='3'*64, packed='4'*64, initial='5'*64, differs=True,
            guards_preserved=True, operands_preserved=True)
            for m in [8, 9, 12, 13, 14, 16, 17, 20, 24, 25] for n in [1024, 1025] for k in [32, 64]]
        self.assertEqual(probe(value, 'normal', '10.0.8', True)['prepared_precedence_cases'], 40)
        for mutate in [lambda r: r['prepared_precedence'].pop(),
            lambda r: r['prepared_precedence'][0].update(candidate='f'*64),
            lambda r: r.update(core_sha256='f'*64), lambda r: r.update(avx512=False),
            lambda r: [v.update(partial='0'*64, differs=False) for v in r['prepared_precedence']]]:
            damaged = copy.deepcopy(value); mutate(damaged)
            with self.assertRaises(AssertionError): probe(damaged, 'normal', '10.0.8', True)

    def test_corpus_equal_process_weighting_keeps_every_clock(self):
        from audit_results import timing_table
        from fractions import Fraction
        manifest = read(OLD / 'manifests/production-parakeet.json')
        results = []
        for process in range(6):
            rows = [dict(name=case['name'], phase='measured', start_ticks=1,
                end_ticks=1+(100*process+i+1)*(j+1), frequency=17, seconds=-1)
                for j in range(3) for i, case in enumerate(manifest['cases'])]
            results.append(dict(records=rows))
        table = timing_table(results, manifest); self.assertEqual(len(table), 21)
        corpus = table[-1]; self.assertEqual(corpus['audio_seconds'], 213.265)
        for role in (*ROLES, 'ort'):
            for process in corpus[role]['processes']:
                index = process['index']; self.assertEqual(TIMING_ROLES[index], role)
                expected = sum(Fraction(r['end_ticks']-r['start_ticks'], r['frequency']) for r in results[index]['records'])/3
                self.assertEqual(Fraction(**process['exact_mean']), expected)

    def test_cache_cleanup_preserves_inputs_and_built_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'campaign').mkdir(); (base/'work/packages').mkdir(parents=True)
            (base/'work/packages/cache.bin').write_bytes(b'disposable')
            (base/'source/bin').mkdir(parents=True); (base/'source/obj').mkdir()
            (base/'source/obj/build.tmp').write_bytes(b'disposable')
            (base/'source/input.cs').write_text('source'); (base/'source/bin/product.dll').write_bytes(b'product')
            (base/'payload.json').write_text(json.dumps(dict(files={'source/input.cs': {}})))
            (base/'campaign/built-files.json').write_text(json.dumps({'source/bin/product.dll': {}}))
            build_caches(base)
            self.assertTrue((base/'source/input.cs').exists() and (base/'source/bin/product.dll').exists())
            self.assertFalse((base/'work/packages').exists() or (base/'source/obj').exists())
            self.assertEqual(read(base/'campaign/build-cache-cleanup.json')['bytes'], 20)

    def test_cache_cleanup_refuses_pinned_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp); (base/'campaign').mkdir(); (base/'work/packages').mkdir(parents=True)
            (base/'work/packages/input').write_bytes(b'pinned')
            (base/'payload.json').write_text(json.dumps(dict(files={'work/packages/input': {}})))
            (base/'campaign/built-files.json').write_text('{}')
            with self.assertRaises(AssertionError): build_caches(base)
            self.assertTrue((base/'work/packages/input').exists())


if __name__ == '__main__': unittest.main()
