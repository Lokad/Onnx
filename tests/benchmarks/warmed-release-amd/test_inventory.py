import copy,json,unittest
from checks import consumer_inventory

def body(new):
    instructions=[dict(offset=0,opcode='ldc.i4' if new else 'ldc.i4.s',operand='0C030000' if new else '78'),
        dict(offset=5 if new else 2,opcode='ldc.i4' if new else 'ldc.i4.s',operand='58020000' if new else '3C'),
        dict(offset=10 if new else 4,opcode='br.s',operand='00'),dict(offset=12 if new else 6,opcode='ret',operand='')]
    return dict(InitLocals=True,MaxStackSize=2,locals=[],exceptions=[],instructions=instructions)

def fixture():
    key='Program::<Main>$::Void <Main>$(System.String[])'
    row=dict(assembly='ReleaseBenchmark.dll',public_surface_equal=True,compiler_rename=None,added=[],removed=[],
        before_sha256='old',after_sha256='new',method_flags_before={key:0},method_flags_after={key:0},
        differences=[key],normalized_methods={key:json.dumps(body(False))},candidate_methods={key:json.dumps(body(True))},
        methods=1,unchanged_methods=0)
    return dict(inventory_complete=True,observations=[row]),dict(previous_consumer=dict(sha256='old')),dict(consumer=dict(sha256='new'))

class InventoryTests(unittest.TestCase):
    def test_exact_count_changes_and_moved_branch_target(self):
        result=consumer_inventory(*fixture());self.assertTrue(result['branches_locals_exceptions_equal'])
        self.assertEqual([(x['before'],x['after']) for x in result['changes']],[(120,780),(60,600)])
    def test_arithmetic_or_branch_change_rejected(self):
        for index,field,wanted in [(0,'operand','0D030000'),(2,'operand','F4')]:
            value,spec,built=fixture();row=value['observations'][0];key=row['differences'][0]
            actual=json.loads(row['candidate_methods'][key]);actual['instructions'][index][field]=wanted
            row['candidate_methods'][key]=json.dumps(actual)
            with self.assertRaises(AssertionError):consumer_inventory(value,spec,built)
    def test_flag_or_extra_method_change_rejected(self):
        value,spec,built=fixture();row=value['observations'][0];key=row['differences'][0]
        row['method_flags_after'][key]=512
        with self.assertRaises(AssertionError):consumer_inventory(value,spec,built)
        row['method_flags_after'][key]=0;row['added']=['new-method']
        with self.assertRaises(AssertionError):consumer_inventory(value,spec,built)

if __name__=='__main__':unittest.main()
