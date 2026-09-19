"""Execute the actual supervisor's pre-launch gate without creating any worker."""
from pathlib import Path
import ast,hashlib,json,os,sys,tempfile,types,unittest
from unittest import mock

SOURCE=Path(__file__).with_name('run.py').read_text(encoding='utf-8')
TREE=ast.parse(SOURCE)
END=next(i for i,node in enumerate(TREE.body) if isinstance(node,ast.Expr) and ast.unparse(node.value)=='out.mkdir()')
PREFIX=compile(ast.Module(body=TREE.body[:END],type_ignores=[]),'frozen-supervisor-gate','exec')

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

class Gate(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup);self.base=Path(self.directory.name)
        (self.base/'bundle.json').write_text('{}',encoding='utf-8');(self.base/'result-control').mkdir()
        (self.base/'result-control/identity.json').write_text(json.dumps(dict(complete=True,supervisor=99999999,runs=[dict(pid=99999900+i,code=0) for i in range(8)])),encoding='utf-8')
        (self.base/'complete-control.txt').write_text('0\n',encoding='utf-8')
        self.gate=dict(phase='control',passed=True,integrity_passed=True,bundle_manifest_sha256=sha(self.base/'bundle.json'),identity_sha256=sha(self.base/'result-control/identity.json'))
    def execute(self,change=None,hash_override=None):
        gate=self.gate.copy();gate.update(change or {})
        p=self.base/'control-audit.json';p.write_text(json.dumps(gate),encoding='utf-8')
        argv=['run.py','compare',hash_override or sha(p)]
        original_iterdir=Path.iterdir
        with mock.patch.object(sys,'argv',argv),mock.patch.object(os,'sched_setaffinity',create=True),mock.patch.dict(sys.modules,{'campaign_processes':types.ModuleType('campaign_processes')}),mock.patch.object(Path,'iterdir',lambda p:iter(()) if p.as_posix()=='/proc' else original_iterdir(p)):
            exec(PREFIX,{'__file__':str(self.base/'run.py'),'__name__':'gate_test'})
        self.assertFalse((self.base/'result-compare').exists())
    def test_valid_receipt_passes_gate_without_worker(self):self.execute()
    def test_failed_or_mismatched_receipts_refuse(self):
        for change in ({'phase':'compare'},{'passed':False},{'integrity_passed':False},{'bundle_manifest_sha256':'wrong'},{'identity_sha256':'wrong'}):
            with self.subTest(change=change):
                with self.assertRaises(AssertionError):self.execute(change)
                self.assertFalse((self.base/'result-compare').exists())
    def test_wrong_transferred_hash_refuses(self):
        with self.assertRaises(AssertionError):self.execute(hash_override='wrong')
    def test_nonzero_control_completion_refuses(self):
        (self.base/'complete-control.txt').write_text('2\n',encoding='utf-8')
        with self.assertRaises(AssertionError):self.execute()
    def test_live_control_supervisor_refuses(self):
        original=Path.exists
        with mock.patch.object(Path,'exists',lambda p:True if p.as_posix().endswith('/proc/99999999') else original(p)):
            with self.assertRaises(AssertionError):self.execute()
    def test_live_control_worker_refuses(self):
        original=Path.exists
        with mock.patch.object(Path,'exists',lambda p:True if p.as_posix().endswith('/proc/99999900') else original(p)):
            with self.assertRaises(AssertionError):self.execute()

if __name__=='__main__':unittest.main()
