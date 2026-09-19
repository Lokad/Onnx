import copy,unittest
from code_audit import pointer_gate

def fixture():
    return dict(fma=24,broadcasts=12,vector_stack=[],calls=[],hot_stack=[],movsxd=1,lea=1,
        hot_loop='\n'.join(f'vbroadcastss zmm26, dword ptr [r8{("+"+hex(i)) if i else ""}]' for i in range(0,48,4))+'\nadd r8, 48\n')

class CodeTests(unittest.TestCase):
    def test_literal_offsets(self):self.assertTrue(pointer_gate(fixture())['passed'])
    def test_damaged_operands(self):
        for old,new in [('[r8+0x4]','[r8+4*r9]'),('[r8+0x4]','[r8+0x8]'),('[r8+0x4]','[r9+0x4]'),('add r8, 48','add r8, 44'),('add r8, 48','add r9, 48')]:
            row=fixture();row['hot_loop']=row['hot_loop'].replace(old,new)
            with self.assertRaises(AssertionError):pointer_gate(row)
    def test_counts_spills_calls(self):
        for key,value in [('fma',23),('broadcasts',11),('movsxd',3),('lea',3),('vector_stack',['spill']),('hot_stack',['load']),('calls',['call'])]:
            row=fixture();row[key]=value
            with self.assertRaises(AssertionError):pointer_gate(row)

if __name__=='__main__':unittest.main()
