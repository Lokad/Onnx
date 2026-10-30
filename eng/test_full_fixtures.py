"""Full fixture preparation retains every workload and never rewrites source fixtures."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import campaign_evidence as evidence

spec=importlib.util.spec_from_file_location('prepare_full_fixtures',Path(__file__).with_name('prepare-full-fixtures.py'))
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


class FullFixtureTests(unittest.TestCase):
    def test_short_schedule_preserves_cases_and_expected_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);assets=root/'assets';assets.mkdir()
            model=assets/'model.onnx';model.write_bytes(b'fixture model')
            values=assets/'values.f32';values.write_bytes(b'1234')
            graphs=assets/'graphs.json'
            graphs.write_text(json.dumps(dict(core='a'*64,cases=[dict(key=name,model='model.onnx',
                inputs=[dict(name='x',shape=[1],file='values.f32')],outputs=[dict(name='y',shape=[1],file='values.f32')])
                for name in ('e5-30tok','dinov3','resnet50','gpt2')])))
            gemma=assets/'gemma';gemma.mkdir();(gemma/'fixtures.json').write_text('{}')
            original={}
            for family,count in [('parakeet',20),('pyannote',4)]:
                value=dict(schema=2,protocol='audio-sustained-v2',family=family,conditioning_passes=6,measured_passes=6,
                    process_timeout_seconds=1800,native_versions=dict(numpy='2.2.4',onnxruntime='1.29.0'),
                    models={'model':{'path':'frozen-model','sha256':'c'*64}},
                    cases=[dict(name=f'{family}-{i}',samples=480000 if i==0 else 16000,
                                raw_sha256='d'*64,expected={'unchanged':i}) for i in range(count)])
                (assets/(family+'.json')).write_text(json.dumps(value));original[family]=value
            before={p:p.read_bytes() for p in assets.rglob('*') if p.is_file()}
            args=SimpleNamespace(prepared=root/'prepared',graphs=graphs,gemma=gemma/'fixtures.json',
                parakeet=assets/'parakeet.json',pyannote=assets/'pyannote.json',audio_root=assets,gemma_root=assets,
                output=root/'full',inventory=None)
            with mock.patch.object(prepare.common,'read_build',return_value={'core_sha256':'b'*64}), \
                    mock.patch.object(prepare.gemma,'fixture',return_value={}):
                prepare.prepare(args)
            config=json.loads((args.output/'full.json').read_text())
            for family in ('parakeet','pyannote'):
                value=json.loads((args.output/(family+'.json')).read_text())
                self.assertEqual(original[family]['cases'],value['cases'])
                self.assertEqual(original[family]['models'],value['models'])
                self.assertEqual((2,3),(value['conditioning_passes'],value['measured_passes']))
                self.assertEqual(evidence.sha256(args.output/(family+'.json')),config['audio'][family]['sha256'])
            current=json.loads((args.output/'graphs.json').read_text())
            self.assertEqual('b'*64,current['core'])
            self.assertEqual(4,len(current['cases']))
            self.assertTrue(all(Path(item['model']).is_absolute() for item in current['cases']))
            self.assertEqual(before,{p:p.read_bytes() for p in assets.rglob('*') if p.is_file()})


if __name__=='__main__':unittest.main()
