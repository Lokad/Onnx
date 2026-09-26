"""The negative probes must reject the intended identity before producing output."""
from pathlib import Path
import tempfile
import unittest
from identity_probes import CASES,command_for,review
from protocol import pin,read,save


class IdentityProbes(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name);(self.base/'identity-probes').mkdir()
        self.spec=dict(identities={});rows=[]
        for role in ['selected','candidate']:
            runtime=self.base/'runtimes'/role;runtime.mkdir(parents=True)
            for name in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll','GraphQualification.dll']:
                (runtime/name).write_bytes((name if name=='GraphQualification.dll' else role+name).encode())
            self.spec['identities'][role]={n:pin(runtime/n) for n in ['Lokad.Onnx.dll','Lokad.Onnx.Data.dll']}
        consumer=pin(self.base/'runtimes/selected/GraphQualification.dll')
        save(self.base/'built.json',dict(consumer=consumer))
        for i,(role,guard) in enumerate(CASES):
            row=dict(role=role,guard=guard,command=command_for(self.spec,role,guard),consumer=consumer,
                     child=dict(pid=i+1,birth=100),terminal=True,code=-6,timed_out=False,
                     seconds=.2,affinity=[2],output_created=False)
            for stream in ['stdout','stderr']:
                path=self.base/f'identity-probes/{role}-{guard}.{stream}'
                path.write_text('' if stream=='stdout' else 'Unhandled exception. System.IO.InvalidDataException: Qualified '+guard+'\n')
                row[stream]=pin(path)
            rows.append(row)
        self.path=self.base/'identity-probes/probes.json';save(self.path,rows)

    def test_all_four_guards(self):
        self.assertEqual(review(self.base,self.spec)['probes'],4)

    def test_successful_process_is_not_a_rejection(self):
        rows=read(self.path);rows[0]['code']=0;save(self.path,rows)
        with self.assertRaises(AssertionError):review(self.base,self.spec)

    def test_wrong_guard_is_rejected_even_with_rehashed_log(self):
        rows=read(self.path);path=self.base/'identity-probes/selected-core.stderr'
        path.write_text('Unhandled exception. System.IO.InvalidDataException: Qualified data\n')
        rows[0]['stderr']=pin(path);save(self.path,rows)
        with self.assertRaises(AssertionError):review(self.base,self.spec)

    def test_output_creation_is_rejected(self):
        (self.base/'identity-probes/selected-core-output').mkdir()
        with self.assertRaises(AssertionError):review(self.base,self.spec)

    def test_unmodified_identity_argument_is_rejected(self):
        rows=read(self.path);rows[0]['command'][5]=self.spec['identities']['selected']['Lokad.Onnx.dll']['sha256'];save(self.path,rows)
        with self.assertRaises(AssertionError):review(self.base,self.spec)


if __name__=='__main__':unittest.main()
