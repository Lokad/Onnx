"""Exercise release rejection paths locally; fixtures are not future qualification."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET
from source_scope import ROOT, QUALIFIED, verify_source, root_files, SOURCE, FIXTURE, FIXTURE_PIN
from checks import inventory, suite, suite256
from protocol import read, pin
from new_cases import NEW_CASES, NEW_SKIPS
from consumer_scope import verify_scope
from prerequisites import validate
from prepare import PRIOR, CONTRACTS, COMPATIBLE, BUILD, gates


class CompiledScope(unittest.TestCase):
    def fixture(self):
        # Construct a same-product comparator from retained real method maps.
        # Only an actual future root build can establish release equivalence.
        value = read(QUALIFIED/'collected/inventory/instructions.json')
        candidate = read(BUILD/'build-collected/logs/instructions.json')
        product = read(COMPATIBLE)['candidate']
        for row, updated in zip(value['observations'], candidate['observations'], strict=True):
            methods = dict(updated['normalized_methods']); methods.update(updated['candidate_methods'])
            row.update(normalized_methods=methods, candidate_methods={}, methods=len(methods),
                unchanged_methods=len(methods), differences=[], added=[], removed=[],
                before_sha256=product[row['assembly']]['sha256'],
                after_sha256=product[row['assembly']]['sha256'],
                method_flags_before=updated['method_flags_after'],
                method_flags_after=copy.deepcopy(updated['method_flags_after']), public_surface_equal=True,
                public_surface=copy.deepcopy(row['public_surface_after']),
                assembly_attributes_before=copy.deepcopy(row['assembly_attributes_after']))
        core = value['observations'][0]
        key, = [k for k in core['normalized_methods'] if '::RunBatchedFloatMatMul::' in k]
        value['release'].update(sha256=product['Lokad.Onnx.dll']['sha256'],
            methods={key: core['normalized_methods'][key]})
        return value, product, copy.deepcopy(product)

    def test_complete_compiled_candidate(self):
        proof = inventory(*self.fixture())
        self.assertEqual(proof['core_methods'], 3288)
        self.assertTrue(proof['public_surface_equal'])

    def test_public_attributes_flags_and_methods_cannot_change(self):
        for mutation in ['public', 'attribute', 'flags', 'body', 'missing-method', 'sha']:
            with self.subTest(mutation=mutation):
                value, measured, built = self.fixture(); row = value['observations'][0]
                if mutation == 'public': row['public_surface_after'] += ['unexpected']
                elif mutation == 'attribute': row['assembly_attributes_after'] += ['unexpected']
                elif mutation == 'flags': row['method_flags_after'].pop(next(iter(row['method_flags_after'])))
                elif mutation == 'body': row['differences'] = ['unexpected']
                elif mutation == 'missing-method':
                    key, = [k for k in row['normalized_methods'] if '::PrepareOwnedMatMulWeights::' in k]
                    row['normalized_methods'].pop(key)
                else: row['after_sha256'] = '0'*64
                with self.assertRaises(AssertionError): inventory(value, measured, built)


class ExactCensus(unittest.TestCase):
    def fixture(self, name, disabled):
        doc = ET.parse(QUALIFIED/'collected'/(name+'-tests'+('-256' if disabled else ''))/(name+'.trx'))
        results = doc.find('.//{*}Results'); template = doc.find('.//{*}UnitTestResult')
        for cases, outcome in [(NEW_CASES, 'Passed'), (NEW_SKIPS, 'NotExecuted')]:
            for case in cases[name]:
                row = copy.deepcopy(template); row.attrib.update(testName=case, outcome=outcome); results.append(row)
        return doc

    def check(self, doc, name, disabled):
        rows = doc.findall('.//{*}UnitTestResult')
        doc.find('.//{*}Counters').attrib.update(total=str(len(rows)),
            passed=str(sum(r.attrib['outcome']=='Passed' for r in rows)),
            failed=str(sum(r.attrib['outcome']=='Failed' for r in rows)))
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); path = folder/'result.trx'; doc.write(path)
            for key in ['backend', 'tensors']:
                for mode in ['', '-256']:
                    (folder/('selected-'+key+mode+'.trx')).write_bytes(
                        (QUALIFIED/'collected'/(key+'-tests'+mode)/(key+'.trx')).read_bytes())
            return (suite256 if disabled else suite)(path, name, folder)

    def test_both_complete_modes(self):
        for name in ['backend', 'tensors']:
            for disabled in [False, True]:
                with self.subTest(name=name, disabled=disabled):
                    proof = self.check(self.fixture(name, disabled), name, disabled)
                    wanted = (3513,133) if name=='backend' and disabled else (3603,43) if name=='backend' else (394,0)
                    self.assertEqual((proof['passed'], proof['skipped']), wanted)

    def test_missing_renamed_skipped_duplicate_or_failed_transpose(self):
        for disabled in [False, True]:
            for mutation in ['missing', 'renamed', 'skipped', 'duplicate', 'failed']:
                with self.subTest(disabled=disabled, mutation=mutation):
                    doc = self.fixture('backend', disabled); results = doc.find('.//{*}Results')
                    row = next(r for r in results if r.attrib.get('testName') == NEW_CASES['backend'][0])
                    if mutation == 'missing': results.remove(row)
                    elif mutation == 'renamed': row.attrib['testName'] += ' changed'
                    elif mutation == 'skipped': row.attrib['outcome'] = 'NotExecuted'
                    elif mutation == 'duplicate': results.append(copy.deepcopy(row))
                    else: row.attrib['outcome'] = 'Failed'
                    with self.assertRaises(AssertionError): self.check(doc, 'backend', disabled)

    def test_existing_disabled_fma_contract_has_exact_expected_outcome(self):
        for disabled in [False, True]:
            for mutation in ['missing', 'passed', 'failed']:
                with self.subTest(disabled=disabled, mutation=mutation):
                    doc = self.fixture('backend', disabled); results = doc.find('.//{*}Results')
                    row = next(r for r in results if r.attrib.get('testName') == 'Lokad.Onnx.Backend.Tests.OwnedAttentionUnavailableTests.DisabledHardwareLeavesSquareAttentionDense')
                    if mutation == 'missing': results.remove(row)
                    else: row.attrib['outcome'] = mutation.title()
                    with self.assertRaises(AssertionError): self.check(doc, 'backend', disabled)

    def test_existing_disabled_hardware_cannot_disappear(self):
        doc = self.fixture('backend', True); results = doc.find('.//{*}Results')
        row = next(r for r in results if r.attrib.get('outcome') == 'NotExecuted'); results.remove(row)
        with self.assertRaises(AssertionError): self.check(doc, 'backend', True)


class Admission(unittest.TestCase):
    def fixture(self):
        reports = {k: read(p/'analysis.json') for k,p in PRIOR.items() if k != 'pyannote-app'}
        # Reuse old timing data ONLY to exercise schema rejection before the new
        # comparison completes. These substituted identities must never be saved
        # or treated as admission of the candidate's Pyannote performance.
        reports['pyannote-app'] = read(ROOT/'artifacts/parakeet-attention-owned-pyannote-app-amd-20260928/analysis.json')
        pair = copy.deepcopy(reports['models']['identities'])
        reports['pyannote-app']['identities'] = copy.deepcopy(pair)
        spec = dict(identities=pair, consumers=reports['baseline']['consumers'], source_prepared=pin(SOURCE/'prepared.json'))
        return reports, spec, read(COMPATIBLE), read(CONTRACTS/'analysis.json'), read(BUILD/'build-review.json')

    def test_admission_schema(self): self.assertTrue(validate(*self.fixture()))

    def test_wrong_identity_or_incomplete_meetings_rejected(self):
        for mutation in ['identity', 'meetings', 'requests', 'gate', 'build', 'source']:
            with self.subTest(mutation=mutation):
                args = self.fixture(); reports,spec,compatible,contracts,build = args
                app = reports['pyannote-app']
                if mutation == 'identity': app['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256'] = '0'*64
                elif mutation == 'meetings': app['results']['meetings-run']['complete_selected_results_exact'] = False
                elif mutation == 'requests': app['timing_requests'] -= 1
                elif mutation == 'gate': app['performance']['admitted'] = False
                elif mutation == 'build': build['methods'][0]['changed'].append('unrelated change')
                else: spec['source_prepared'] = {'sha256': 'unreviewed'}
                with self.assertRaises(AssertionError): validate(*args)

    def test_focused_contract_modes_names_and_review_cannot_change(self):
        for mutation in ['missing-mode', 'count', 'classes', 'missing-name', 'duplicate-name', 'review', 'arithmetic', 'data', 'release']:
            with self.subTest(mutation=mutation):
                args = self.fixture(); contracts,build = args[3:5]
                if mutation == 'missing-mode': contracts['suites'].pop()
                elif mutation == 'count': contracts['suites'][0]['passed'] -= 1
                elif mutation == 'classes': contracts['suites'][0]['classes']['TransposeAxisMovementTests'] -= 1
                elif mutation == 'missing-name': contracts['suites'][2]['names'].pop()
                elif mutation == 'duplicate-name': contracts['suites'][0]['names'][0] = contracts['suites'][0]['names'][1]
                elif mutation == 'review': contracts['compiled_review']['sha256'] = '0'*64
                elif mutation == 'arithmetic': build['arithmetic_leaves_unchanged'] = False
                elif mutation == 'data': build['data_binary_unchanged'] = False
                else: contracts['release_admitted'] = True
                with self.assertRaises(AssertionError): validate(*args)



class Source(unittest.TestCase):
    def test_exact_source_and_reused_workflow(self):
        self.assertEqual(len(root_files(verify_source())), 447)
        self.assertTrue(verify_scope())

    def test_fixture_cases_match_executed_modes(self):
        self.assertEqual(pin(FIXTURE), FIXTURE_PIN)
        focused = read(CONTRACTS/'analysis.json')['suites']
        for row in focused:
            names = [n for n in row['names'] if '.TransposeAxisMovementTests.' in n]
            self.assertEqual(set(names), set(NEW_CASES[row['suite']]))
        self.assertEqual((len(NEW_CASES['backend']), len(NEW_SKIPS['backend'])), (6,0))

    def test_either_pending_comparison_prevents_source_application(self):
        for graph, app in [(None,None), (None,'0'*64), ('0'*64,None)]:
            with self.subTest(graph=graph, app=app):
                with patch('prepare.GRAPH_DIGEST', graph), patch('prepare.PYANNOTE_APP_DIGEST', app):
                    with self.assertRaisesRegex(AssertionError, 'Require actual admitted graph and complete Pyannote application closures'):
                        gates()


if __name__ == '__main__': unittest.main()
