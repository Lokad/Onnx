"""Damage retained maximum-speech decisions and resource evidence; require refusal."""
from pathlib import Path
import copy
import json
import unittest
from tokenizers import Tokenizer
import audit
from prepare import read, sha


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[3]
        cls.base = root/'artifacts/whisper-maximum-speech-20260919'
        cls.inputs = read(cls.base/'inputs/inputs.json')
        cls.native = read(cls.base/'native/manifest.json')
        cls.short = read(cls.base/'short/manifest.json')
        cls.validator = audit.validator_at(cls.base/'reference')
        cls.tokenizer = Tokenizer.from_file(str(root/'models/whisper-large-v3-turbo/tokenizer.json'))
        cls.original = read(root/'artifacts/whisper-recording-v2-20260919/managed/result.json')

    def fixture(self):
        managed = copy.deepcopy(self.original)
        managed['inputs_sha256'] = self.native['inputs_sha256']
        managed['cases'] = [dict(name='maximum-speech',repeat=repeat,seconds=1.,ownership=True,
            pcm_sha256=self.inputs['cases'][0]['pcm_sha256'],result=copy.deepcopy(self.native['cases'][0]['result'])) for repeat in (False,True)]
        return managed

    def app(self,managed):
        return audit.application(managed,self.native,self.inputs,self.short,self.validator,self.tokenizer)

    def test_real_oracle_fixture_and_damaged_application(self):
        self.assertEqual(len(self.app(self.fixture())),2)
        mutations = [
            lambda j:j['cases'].pop(),
            lambda j:j['cases'][0].update(repeat=True),
            lambda j:j['cases'][0].update(seconds=float('nan')),
            lambda j:j['cases'][0].update(pcm_sha256='wrong'),
            lambda j:j['cases'][0].update(ownership=False),
            lambda j:j.update(refusals=9),
            lambda j:j.update(flags={'DOTNET_TieredCompilation':'0'}),
            lambda j:j['silent']['windows'].pop(),
            lambda j:j['concurrent'].pop(),
            lambda j:j['short_regression']['token_ids'].append(1),
            lambda j:[c['result'].update(processed_seconds=599) for c in j['cases']],
            lambda j:[c['result']['windows'][0].update(advanced_seconds=0) for c in j['cases']],
            lambda j:[c['result']['windows'][0]['decoding'].update(average_log_probability=float('nan')) for c in j['cases']],
            lambda j:[c['result']['segments'][0].update(text='incorrect') for c in j['cases']],
            lambda j:[c['result']['windows'][0]['decoding']['token_ids'].__setitem__(0,True) for c in j['cases']],
        ]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                damaged = self.fixture()
                mutate(damaged)
                with self.assertRaises((AssertionError,ValueError,KeyError)):
                    self.app(damaged)

    def test_confidence_is_a_diagnostic(self):
        changed = self.fixture()
        for case in changed['cases']:
            case['result']['windows'][0]['decoding']['no_speech_probability'] = .5
        self.assertGreater(self.app(changed)[0]['maximum_native_confidence_difference'],.01)

    def test_retained_resource_refusals(self):
        identity = read(self.base/'native-process.json')
        samples = [json.loads(line) for line in (self.base/'native-samples.jsonl').read_text(encoding='utf-8').splitlines()]
        digest = sha(self.base/'frozen.json')
        recovery = read(self.base/'native-terminal-recovery.json') if (self.base/'native-terminal-recovery.json').exists() else None
        self.assertGreater(audit.process(identity,samples,'native',digest,recovery)['samples'],1)
        mutations = [
            lambda j,s:j.update(complete=not identity['complete']),
            lambda j,s:j.update(code=1),
            lambda j,s:j.update(samples=0),
            lambda j,s:j.update(peak_rss=0),
            lambda j,s:j.update(frozen_sha256='wrong'),
            lambda j,s:j['limits'].update(seconds=8000),
            lambda j,s:s[0].update(available_memory=0),
            lambda j,s:s[0].update(rss=16*1024**3),
            lambda j,s:s[-1].update(seconds=-1),
            lambda j,s:s[1]['members'][0].update(affinity=[0,2]),
            lambda j,s:s[1]['members'][0].update(create_time=0),
        ]
        for index,mutate in enumerate(mutations):
            with self.subTest(index=index):
                damaged,values = copy.deepcopy(identity),copy.deepcopy(samples)
                mutate(damaged,values)
                with self.assertRaises((AssertionError,ValueError,KeyError)):
                    audit.process(damaged,values,'native',digest,recovery)


if __name__ == '__main__':
    unittest.main()
