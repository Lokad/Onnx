"""Reject changed products, missing tests and failed admission without running a model."""
import copy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET
from source_scope import ROOT,QUALIFIED,TOOLS,verify_source,root_files,SOURCE,TEST,FIXTURE,load
from checks import inventory,suite,suite256
from protocol import read,pin
from new_cases import NEW_CASES
from consumer_scope import verify_scope
from prerequisites import validate
from prepare import PRIOR,CONTRACTS,COMPATIBLE


class CompiledScope(unittest.TestCase):
    def fixture(self):
        # Construct a same-product comparator result from retained real method maps.
        # This is a checker fixture; it is not proof of a future root build.
        value=read(QUALIFIED/'collected/inventory/instructions.json')
        pad=read(CONTRACTS/'build-collected/logs/instructions.json')
        product=read(CONTRACTS/'analysis.json')['product']
        for row,updated in zip(value['observations'],pad['observations'],strict=True):
            methods=dict(updated['normalized_methods']);methods.update(updated['candidate_methods'])
            row.update(normalized_methods=methods,candidate_methods={},methods=len(methods),unchanged_methods=len(methods),
                       differences=[],added=[],removed=[],before_sha256=product[row['assembly']]['sha256'],
                       after_sha256=product[row['assembly']]['sha256'],method_flags_before=updated['method_flags_after'],
                       method_flags_after=copy.deepcopy(updated['method_flags_after']),public_surface_equal=True,
                       public_surface=row['public_surface_after'],assembly_attributes_before=row['assembly_attributes_after'])
        core=value['observations'][0]
        key,=[k for k in core['normalized_methods'] if '::RunBatchedFloatMatMul::' in k]
        value['release'].update(sha256=product['Lokad.Onnx.dll']['sha256'],methods={key:core['normalized_methods'][key]})
        return value,product,copy.deepcopy(product)

    def test_complete_compiled_candidate(self):
        proof=inventory(*self.fixture());self.assertEqual(proof['core_methods'],3283)
        self.assertTrue(proof['public_surface_equal'])

    def test_public_attributes_flags_and_methods_cannot_change(self):
        mutations=['public','attribute','flags','body','missing-helper','sha']
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                value,measured,built=self.fixture();row=value['observations'][0]
                if mutation=='public':row['public_surface_after']=row['public_surface_after']+['unexpected']
                elif mutation=='attribute':row['assembly_attributes_after']=row['assembly_attributes_after']+['unexpected']
                elif mutation=='flags':row['method_flags_after'].pop(next(iter(row['method_flags_after'])))
                elif mutation=='body':row['differences']=['unexpected']
                elif mutation=='missing-helper':
                    key,=[k for k in row['normalized_methods'] if '::SigmoidRationalVector::' in k]
                    row['normalized_methods'].pop(key)
                else:row['after_sha256']='0'*64
                with self.assertRaises(AssertionError):inventory(value,measured,built)


class ExactCensus(unittest.TestCase):
    def fixture(self,name,disabled):
        doc=ET.parse(QUALIFIED/'collected'/(name+'-tests'+('-256' if disabled else ''))/(name+'.trx'))
        results=doc.find('.//{*}Results');template=doc.find('.//{*}UnitTestResult')
        for case in NEW_CASES[name]:
            row=copy.deepcopy(template);row.attrib.update(testName=case,outcome='Passed');results.append(row)
        return doc

    def check(self,doc,name,disabled):
        rows=doc.findall('.//{*}UnitTestResult')
        doc.find('.//{*}Counters').attrib.update(total=str(len(rows)),
            passed=str(sum(r.attrib['outcome']=='Passed' for r in rows)),
            failed=str(sum(r.attrib['outcome']=='Failed' for r in rows)))
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp);path=folder/'result.trx';doc.write(path)
            for key in ['backend','tensors']:
                for mode in ['', '-256']:
                    (folder/('selected-'+key+mode+'.trx')).write_bytes(
                        (QUALIFIED/'collected'/(key+'-tests'+mode)/(key+'.trx')).read_bytes())
            return (suite256 if disabled else suite)(path,name,folder)

    def test_both_complete_modes(self):
        for name in ['backend','tensors']:
            for disabled in [False,True]:
                with self.subTest(name=name,disabled=disabled):
                    proof=self.check(self.fixture(name,disabled),name,disabled)
                    wanted=(3470,132) if name=='backend' and disabled else (3560,42) if name=='backend' else (394,0)
                    self.assertEqual((proof['passed'],proof['skipped']),wanted)

    def test_missing_renamed_skipped_duplicate_or_failed_sigmoid(self):
        for disabled in [False,True]:
            for mutation in ['missing','renamed','skipped','duplicate','failed']:
                with self.subTest(disabled=disabled,mutation=mutation):
                    doc=self.fixture('backend',disabled);results=doc.find('.//{*}Results')
                    row=next(r for r in results if r.attrib.get('testName')==NEW_CASES['backend'][0])
                    if mutation=='missing':results.remove(row)
                    elif mutation=='renamed':row.attrib['testName']+=' changed'
                    elif mutation=='skipped':row.attrib['outcome']='NotExecuted'
                    elif mutation=='duplicate':results.append(copy.deepcopy(row))
                    else:row.attrib['outcome']='Failed'
                    with self.assertRaises(AssertionError):self.check(doc,'backend',disabled)

    def test_existing_disabled_hardware_cannot_disappear(self):
        doc=self.fixture('backend',True);results=doc.find('.//{*}Results')
        row=next(r for r in results if r.attrib.get('outcome')=='NotExecuted');results.remove(row)
        with self.assertRaises(AssertionError):self.check(doc,'backend',True)


class Admission(unittest.TestCase):
    def fixture(self):
        reports={k:read(p/'analysis.json') for k,p in PRIOR.items() if k!='pyannote-app'}
        # Retained timing data exercises schema/negative guards only. Never publish
        # these substituted identities as qualification of the new product.
        reports['pyannote-app']=read(ROOT/'artifacts/parakeet-pad-current-pyannote-app-amd-20260926/analysis.json')
        pair=copy.deepcopy(reports['models']['identities'])
        reports['pyannote-app']['identities']=copy.deepcopy(pair)
        spec=dict(identities=pair,consumers=reports['baseline']['consumers'],source_prepared=pin(SOURCE/'prepared.json'))
        return reports,spec,read(COMPATIBLE),read(CONTRACTS/'analysis.json'),read(CONTRACTS/'build-review.json')

    def test_admission_schema(self):self.assertTrue(validate(*self.fixture()))

    def test_wrong_identity_or_incomplete_meetings_rejected(self):
        for mutation in ['identity','meetings','requests','gate','build','source']:
            with self.subTest(mutation=mutation):
                reports,spec,compatible,contracts,compiled=self.fixture();app=reports['pyannote-app']
                if mutation=='identity':app['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256']='0'*64
                elif mutation=='meetings':app['results']['meetings-run']['complete_selected_results_exact']=False
                elif mutation=='requests':app['timing_requests']-=1
                elif mutation=='gate':app['performance']['admitted']=False
                elif mutation=='build':compiled['methods'][0]['changed'].append('unrelated change')
                else:spec['source_prepared']={'sha256':'unreviewed'}
                with self.assertRaises(AssertionError):validate(reports,spec,compatible,contracts,compiled)


class Source(unittest.TestCase):
    def test_exact_source_and_reused_workflow(self):
        self.assertEqual(len(root_files(verify_source())),439);self.assertTrue(verify_scope())

    def test_fixture_preserves_arithmetic_facts(self):
        module=load('portable_fixture_test',TOOLS.parent/'rational-sigmoid-integration/prepare_tests.py')
        expected,removed=module.corrected_fixture((SOURCE/'source'/TEST).read_text())
        self.assertEqual(expected,(FIXTURE/'SigmoidVectorTests.cs').read_text())
        self.assertEqual(removed,(FIXTURE/'runtime-guard-retained.txt').read_text())
        with self.assertRaises(AssertionError):module.corrected_fixture('not the measured fixture')



if __name__=='__main__':unittest.main()
