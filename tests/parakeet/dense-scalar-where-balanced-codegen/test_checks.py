"""Adversarial journal and timing-boundary checks, using synthetic data only."""
import copy
import unittest
from checks import inspect_il,check_journal,expected_journal


class Checks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows=[dict(clocks=[dict(index=i,iteration=j,warmup=j<600,ticks=1) for j in range(780)]) for i in range(220)]
        cls.journal=list(expected_journal(cls.rows))

    def test_valid_journal(self):check_journal(self.journal,self.rows)

    def test_missing(self):
        with self.assertRaises(AssertionError):check_journal(self.journal[:-1],self.rows)

    def test_duplicate(self):
        values=list(self.journal);values[1]=values[0]
        with self.assertRaises(AssertionError):check_journal(values,self.rows)

    def test_reordered(self):
        values=list(self.journal);values[0],values[1]=values[1],values[0]
        with self.assertRaises(AssertionError):check_journal(values,self.rows)

    def test_wrong_phase(self):
        values=list(self.journal);values[132000]=dict(values[132000],warmup=True)
        with self.assertRaises(AssertionError):check_journal(values,self.rows)

    def test_old_contiguous_order(self):
        values=[c for row in self.rows for c in row['clocks'][:600]]+[c for row in self.rows for c in row['clocks'][600:]]
        with self.assertRaises(AssertionError):check_journal(values,self.rows)

    @staticmethod
    def il():
        codes=['call','stloc.0','ldc.i4.0','stloc.1','br.s','ldarg.3','ldloc.1','ldelema','ldarg.0','ldarg.1','ldarg.2','ldnull','call','stobj',
            'ldloc.1','ldc.i4.1','add','stloc.1','ldloc.1','ldarg.3','ldlen','conv.i4','blt.s','call','ldloc.0','sub','ret']
        rows=[dict(offset=i,opcode=c,operand=None) for i,c in enumerate(codes)]
        rows[0]['operand']=rows[-4]['operand']='System.Diagnostics.Stopwatch:Int64 GetTimestamp()'
        rows[12]['operand']='Lokad.Onnx.CPUExecutionProvider:Lokad.Onnx.OpResult Where(Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ExecutionOptions)'
        rows[7]['operand']=rows[13]['operand']='Lokad.Onnx.OpResult'
        rows[4]['operand']=18;rows[-5]['operand']=5
        return dict(passed=True,implementation_flags=8,method='Int64 MeasureBatch(Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.ITensor, Lokad.Onnx.OpResult[])',
            locals=['System.Int64','System.Int32'],exceptions=0,instructions=rows,il_bytes=len(rows))

    def test_valid_il(self):self.assertTrue(inspect_il(self.il())['passed'])

    def test_changed_il(self):
        for change in [lambda v:v.update(implementation_flags=520),lambda v:v['instructions'][12].update(operand='helper'),
                       lambda v:v['instructions'][4].update(operand=5),lambda v:v['instructions'][-5].update(operand=6),
                       lambda v:v.update(exceptions=1),lambda v:v['instructions'][16].update(opcode='sub')]:
            v=self.il();change(v)
            with self.assertRaises(AssertionError):inspect_il(v)


if __name__=='__main__':unittest.main()
