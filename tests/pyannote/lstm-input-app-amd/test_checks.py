"""Reject damaged real native requests and missing completed qualification gates."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
import prepare
from checks import prereqs, qualify, timing_table
from protocol import read, save, pin

ROOT = prepare.ROOT
OLD = prepare.APP/'collected/campaign'


class Checks(unittest.TestCase):
    def test_every_closed_prerequisite_required(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); pins = {}
            for name,(suffix,_) in prepare.PRIOR.items():
                source = ROOT/'artifacts'/('pyannote-lstm-input-'+suffix+'-20260922')
                target = base/'evidence'/name; target.mkdir(parents=True)
                for file in ['closed.json','analysis.json']: shutil.copy2(source/file,target/file)
                pins[name] = dict(closed=pin(target/'closed.json'),analysis=pin(target/'analysis.json'))
            spec = dict(prerequisites=pins,identities=read(base/'evidence/models/analysis.json')['identities'])
            self.assertTrue(prereqs(base,spec)['passed'])
            damaged = copy.deepcopy(spec); del damaged['prerequisites']['shared']
            with self.assertRaises((AssertionError,KeyError)): prereqs(base,damaged)
            path = base/'evidence/product/analysis.json'; value = read(path)
            value['suites']['backend']['passed'] -= 1; save(path,value)
            with self.assertRaises(AssertionError): prereqs(base,spec)

    def test_native_real_requests_and_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); (base/'manifests').mkdir()
            shutil.copytree(prepare.APP_PAYLOAD/'runtime',base/'runtime')
            shutil.copy2(prepare.APP_PAYLOAD/'manifests/production-pyannote.json',base/'manifests/selected-pyannote.json')
            shutil.copytree(OLD/'native-conformance-output',base/'native-pyannote/output')
            spec = read(prepare.APP_PAYLOAD/'payload.json')
            self.assertTrue(qualify(base,'native-pyannote',spec)['passed'])
            path = base/'native-pyannote/output/result.json'; original = read(path)
            for key in ['missing','ownership','native-setting','clock','timeline']:
                damaged = copy.deepcopy(original)
                if key == 'missing': damaged['records'].pop()
                if key == 'ownership': damaged['held_outputs_unchanged'] = False
                if key == 'native-setting': damaged['native_settings']['intra_threads'] = 2
                if key == 'clock': damaged['records'][0]['end_ticks'] += 100000
                if key == 'timeline': damaged['records'][0]['result']['intervals'][0][2] += 1
                save(path,damaged)
                with self.assertRaises(AssertionError): qualify(base,'native-pyannote',spec)

    def test_all_original_timing_clocks_retained(self):
        old_roles = ['production','portable','ort','ort','portable','production']
        results = [read(OLD/f'timing-{i:02}-{role}-output/result.json') for i,role in enumerate(old_roles)]
        table = timing_table(results,read(prepare.APP_PAYLOAD/'manifests/production-pyannote.json'))
        previous = read(prepare.APP/'analysis.json')['table']
        for current,old in zip(table,previous,strict=True):
            self.assertEqual(current['name'],old['name'])
            for role,before in [('selected','production'),('candidate','portable'),('ort','ort')]:
                self.assertEqual(current[role],old[before])
        damaged = copy.deepcopy(results)
        damaged[0]['records'] = [r for r in damaged[0]['records'] if r['pass'] != 3]
        with self.assertRaises(AssertionError): timing_table(damaged,read(prepare.APP_PAYLOAD/'manifests/production-pyannote.json'))
