import copy,unittest
from audit import identity,resources
from common import CORE,FILTER,LIMITS,REMOTE
from code_audit import inspect

class AuditTests(unittest.TestCase):
    def assembly(self):
        product='; Assembly listing for method Lokad.Onnx.Tensor`1[float]:LayerNormFloatInto() (Tier1)\n'
        product+='\n'.join(op+' ymm0, ymm1, ymm2' for op in ['vsubpd','vmulpd','vaddpd'])+'\n; Total bytes of code 100\n'
        wide='; Assembly listing for method LayerNormOutput.Kernels:WideOutput() (FullOpts)\n'
        wide+='\n'.join(op+' zmm0, zmm1, zmm2' for op in ['vsubpd','vmulpd','vaddpd'])
        wide+='\nvcvtps2pd zmm0, ymm1\nvcvtpd2ps ymm0, zmm1\n; Total bytes of code 200\n'
        return product+wide

    def test_code_and_damaged_instructions(self):
        text=self.assembly();self.assertEqual(inspect(text)['wide_code_bytes'],200)
        for damaged in [text.replace('(FullOpts)','(Tier0)'),text.replace('vmulpd zmm','vmulpd ymm'),text.replace('vcvtpd2ps','nop'),
                        text.replace('Kernels:WideOutput','Kernels:Other'),text+'vfmadd231pd zmm0, zmm1, zmm2\n',text.replace('vaddpd ymm','vaddpd zmm')]:
            with self.assertRaises(AssertionError):inspect(damaged)

    def test_runtime_and_declared_flags(self):
        value=dict(mode='proof',core_sha256=CORE,probe_sha256='probe',runtime='10.0.8',affinity=4,vector_width=8,vector512_hardware=True,avx512=True,settings={})
        identity(value,dict(sha256='probe'),'proof')
        for key,want in [('runtime','10.0.12'),('affinity',1),('vector_width',16),('vector512_hardware',False),('avx512',False),('core_sha256','other'),('settings',{'DOTNET_TieredCompilation':'0'})]:
            damaged=copy.deepcopy(value);damaged[key]=want
            with self.subTest(key=key),self.assertRaises(AssertionError):identity(damaged,dict(sha256='probe'),'proof')
        value.update(mode='code',settings=dict(COMPlus_JitDisasm=FILTER,COMPlus_JitStdOutFile=REMOTE+'/result/code/jit.txt'))
        identity(value,dict(sha256='probe'),'code')
        value['settings']['COMPlus_JitDisasm']='*'
        with self.assertRaises(AssertionError):identity(value,dict(sha256='probe'),'code')

    def test_jit_conversion_spelling_and_narrow_source_refusal(self):
        text=self.assembly()
        # Actual .NET 10.0.8 dump uses the instruction size for both operands.
        dumped=text.replace('vcvtps2pd zmm0, ymm1','vcvtps2pd zmm0, zmm1').replace('vcvtpd2ps ymm0, zmm1','vcvtpd2ps zmm0, zmm1')
        self.assertTrue(inspect(dumped)['passed'])
        for original in [text,dumped]:
            for replacement in ['vcvtpd2ps ymm0, ymm1','vcvtpd2ps zmm0, ymm1','vcvtpd2ps xmm0, zmm1']:
                damaged=original.replace('vcvtpd2ps ymm0, zmm1',replacement).replace('vcvtpd2ps zmm0, zmm1',replacement)
                with self.subTest(replacement=replacement),self.assertRaises(AssertionError):inspect(damaged)

    def test_resource_births_and_bounds(self):
        state=dict(code=0,terminal_members=True,seconds=1.,samples=2,started=10.,ended=12.,members={'123':1.},child=dict(pid=123,birth=1.),peak_rss=400)
        rows=[dict(seconds=t,available=4*1024**3,members=[dict(pid=123,birth=1.,affinity=[2],rss=400)]) for t in [.1,.6]]
        self.assertEqual(resources(state,rows)['peak_rss'],400)
        for key,value in [('code',2),('terminal_members',False),('seconds',181),('samples',3),('peak_rss',401)]:
            damaged=copy.deepcopy(state);damaged[key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):resources(damaged,rows)
        for key,value in [('birth',2.),('affinity',[0]),('rss',7*1024**3)]:
            damaged=copy.deepcopy(rows);damaged[0]['members'][0][key]=value
            with self.subTest(key=key),self.assertRaises(AssertionError):resources(state,damaged)

if __name__=='__main__':unittest.main()
