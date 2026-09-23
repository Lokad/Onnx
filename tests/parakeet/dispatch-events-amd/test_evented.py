import json,tempfile,unittest
from pathlib import Path
from audit_evented import evented_inventory
class Evented(unittest.TestCase):
 def setUp(self):
  self.value={'$schema':'https://www.speedscope.app/file-format-schema.json','shared':{'frames':[{'name':'outer'},{'name':'leaf'}]},'profiles':[dict(type='evented',unit='milliseconds',name='Thread (9)',startValue=0,endValue=3,events=[dict(type='O',at=0,frame=0),dict(type='O',at=1,frame=1),dict(type='C',at=2,frame=1),dict(type='C',at=3,frame=0)])]}
 def check(self):
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'speedscope.json';path.write_text(json.dumps(self.value));return evented_inventory(path)
 def reject(self):
  with self.assertRaises(AssertionError):self.check()
 def test_balanced(self):self.assertEqual(self.check()['profiles'][0]['active_ms'],3)
 def test_unbalanced(self):self.value['profiles'][0]['events'].pop();self.reject()
 def test_wrong_close(self):self.value['profiles'][0]['events'][2]['frame']=0;self.reject()
 def test_bad_frame(self):self.value['profiles'][0]['events'][1]['frame']=2;self.reject()
 def test_rounding(self):self.value['profiles'][0]['events'][2]['at']=.9999;self.assertEqual(len(self.check()['rounding_adjustments_ms']),1)
 def test_excessive_rounding(self):self.value['profiles'][0]['events'][2]['at']=.9;self.reject()
 def test_empty_accounting(self):self.value['profiles'][0]['events'][0]['at']=.5;self.assertEqual(self.check()['profiles'][0]['empty_ms'],.5)
if __name__=='__main__':unittest.main()
