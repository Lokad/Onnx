"""Use isolated copies of historical evidence to test complete census and rejection."""
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
import prepare
from protocol import pin, read, save
from checks import qualify


class Checks(unittest.TestCase):
    def test_complete_scope_and_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory); (base/'evidence').mkdir()
            shutil.copytree(prepare.REFERENCE, base/'reference')
            (base/'e5').mkdir()
            for case in prepare.CASES:
                fixture = prepare.E5/(case+'.json'); value = read(fixture)
                for source in [fixture, prepare.E5/value['reference_file']]: shutil.copy2(source, base/'e5'/source.name)
            total_arrays = total_values = 0
            for mode in ['shared','e5']:
                source = prepare.OLD/'outputs'/(mode+'-0-baseline')
                original = read(source/'result.json')
                save(base/'evidence'/(mode+'-historical.json'), original)
                core = dict(sha256=original['core_sha256'])
                spec = dict(identities={role: {'Lokad.Onnx.dll':core} for role in ['selected','candidate']},
                    consumer=dict(sha256=original['probe_sha256']), runtime='10.0.8')
                # Synthetic runtime metadata exists only in this temporary checker fixture.
                value = copy.deepcopy(original); value['runtime'] = '10.0.8'
                for role in ['selected','candidate']:
                    folder = base/(role+'-'+mode)/'output'; shutil.copytree(source, folder)
                    save(folder/'result.json', value)
                    report = qualify(base, role+'-'+mode, spec)
                    self.assertTrue(report['passed'])
                total_arrays += report['arrays']; total_values += report['values']
                path = base/('candidate-'+mode)/'output/result.json'
                for key in ['missing','ownership','flag','shape']:
                    damaged = copy.deepcopy(value)
                    if key == 'missing': damaged['rows'].pop()
                    if key == 'ownership': damaged['held_outputs_unchanged'] = False
                    if key == 'flag': damaged['flags']['DOTNET_EnableAVX512'] = '0'
                    if key == 'shape': damaged['rows'][0]['shape'][0] += 1
                    save(path, damaged)
                    with self.assertRaises(AssertionError): qualify(base, 'candidate-'+mode, spec)
                save(path, value)
                tensor = path.parent/value['rows'][0]['file']
                content = bytearray(tensor.read_bytes()); content[0] ^= 1; tensor.write_bytes(content)
                with self.assertRaises(AssertionError): qualify(base, 'candidate-'+mode, spec)
            self.assertEqual((total_arrays,total_values), (166,5000814))


if __name__ == '__main__': unittest.main()
