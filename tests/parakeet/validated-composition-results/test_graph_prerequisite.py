"""Reject misleading qualified-case summaries even when their own hashes agree."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tests/benchmarks/e5-warmed-qualification-amd'))
from protocol import pin,read,save
from graph_prerequisite import verify_bundle


class Prerequisite(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='test-graph-prereq-',dir=ROOT/'artifacts')
        self.base=Path(self.temporary.name).resolve()
        assert self.base.parent==(ROOT/'artifacts').resolve()
        self.addCleanup(self.cleanup)
        sources={'graphs':'parakeet-validated-composition-graphs-amd-20260924',
            'e5':'e5-warmed-qualification-amd-20260924','graph-qualification':'parakeet-composition-graph-qualification-20260924'}
        for label,folder in sources.items():
            destination=self.base/'evidence'/label;destination.mkdir(parents=True)
            for name in ['closed.json','analysis.json']+([] if label=='graph-qualification' else ['payload.json']):
                shutil.copy2(ROOT/'artifacts'/folder/name,destination/name)
        folder=self.base/'evidence/graph-qualification'
        self.spec=dict(graph_qualification=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json')),
            identities=read(ROOT/'artifacts/parakeet-validated-composition-pyannote-amd-20260924/analysis.json')['identities'])

    def cleanup(self):
        assert self.base.resolve().parent==(ROOT/'artifacts').resolve()
        self.temporary.cleanup()

    def forge_summary(self,edit):
        folder=self.base/'evidence/graph-qualification';value=read(folder/'analysis.json');edit(value)
        save(folder/'analysis.json',value)
        proof=read(folder/'closed.json');proof['files']['analysis.json']=pin(folder/'analysis.json');save(folder/'closed.json',proof)
        self.spec['graph_qualification']=dict(closed=pin(folder/'closed.json'),analysis=pin(folder/'analysis.json'))

    def test_actual_complete_sources_pass(self):self.assertTrue(verify_bundle(self.base,self.spec)['passed'])

    def test_root_measured_product_matches(self):
        self.spec['measured']=self.spec.pop('identities')['candidate']
        self.assertTrue(verify_bundle(self.base,self.spec)['passed'])

    def test_root_measured_product_substitution_rejected(self):
        self.spec['measured']=self.spec.pop('identities')['selected']
        with self.assertRaises(AssertionError):verify_bundle(self.base,self.spec)

    def test_hidden_warmup_change_rejected(self):
        self.forge_summary(lambda value:next(r for r in value['performance'] if r['key']=='e5-30tok').update(warmups=600))
        with self.assertRaises(AssertionError):verify_bundle(self.base,self.spec)

    def test_missing_case_rejected(self):
        self.forge_summary(lambda value:value['performance'].pop())
        with self.assertRaises(AssertionError):verify_bundle(self.base,self.spec)

    def test_cross_product_substitution_rejected(self):
        self.forge_summary(lambda value:value['products']['candidate']['Lokad.Onnx.dll'].update(sha256='0'*64))
        with self.assertRaises(AssertionError):verify_bundle(self.base,self.spec)

    def test_original_failed_e5_cannot_replace_successor(self):
        original=read(self.base/'evidence/graphs/analysis.json')
        failed=next(r for r in original['performance'] if r['key']=='e5-30tok')
        self.forge_summary(lambda value:next(r for r in value['performance'] if r['key']=='e5-30tok').update(failed,qualified=True,regression_passed=True))
        with self.assertRaises(AssertionError):verify_bundle(self.base,self.spec)


if __name__=='__main__':unittest.main()
