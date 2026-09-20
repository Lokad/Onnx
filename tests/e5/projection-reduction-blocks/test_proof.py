import copy, unittest
from proof_common import LIMITS, cases, validate_resources
from code_audit import inspect


def fixture():
    state=dict(complete=True,code=0,limits=LIMITS.copy(),started=10.,ended=20.,supervisor=dict(pid=1,start=100),runs=[]);samples={}
    for i,name in enumerate(['plain','code']):
        pid=i+2;row=dict(name=name,pid=pid,start=200+i,started=11.+i*3,ended=13.+i*3,seconds=2.,code=0,
                         preflight_available=4*1024**3,preflight_disk=128*1024**2,members={str(pid):200+i},samples=1,peak_rss=100)
        state['runs'].append(row)
        samples[name]=[dict(seconds=1.,available_memory=2*1024**3,members=[dict(pid=pid,start=200+i,group=pid,affinity='2',rss=100,cpu_seconds=.5)])]
    return state,samples


def code():
    output=''
    for owner in ['Original','Blocked']:
        for width in [12,8]:
            output+=f'; Assembly listing for method ReductionProbe.{owner}:PackedTile{width}(int) (FullOpts)\nG_M000_IG03:\n'
            output+='       vbroadcastss zmm0, dword ptr [rax]\n'*width
            output+='       vfmadd213ps zmm1, zmm0, zmm2\n'*(width*2)
            output+='       inc rax\n       cmp rax, rdx\n       jl SHORT G_M000_IG03\n; Total bytes of code 500\n'
    return output


class ProofTests(unittest.TestCase):
    def test_coverage(self):
        self.assertEqual(len(cases()),473)
        self.assertEqual(sum(r[3] for r in cases()),10)
        for m in range(8,46):self.assertIn((m,129,96,False),cases())
        for n in [127,128,129,255,256,257]:self.assertIn((128,n,96,False),cases())
        self.assertIn((512,1536,384,False),cases())

    def test_resource_refusals(self):
        state,samples=fixture();self.assertEqual(len(validate_resources(state,samples)['births']),3)
        for change in ['affinity','birth','group','rss','available','time','count','duplicate','exit','order','peak']:
            s,r=copy.deepcopy((state,samples));member=r['plain'][0]['members'][0];run=s['runs'][0]
            if change=='affinity':member['affinity']='0'
            elif change=='birth':member['start']+=1
            elif change=='group':member['group']=7
            elif change=='rss':member['rss']=LIMITS['rss']
            elif change=='available':r['plain'][0]['available_memory']=0
            elif change=='time':r['plain'][0]['seconds']=200
            elif change=='count':run['samples']=2
            elif change=='duplicate':r['plain'][0]['members'].append(member.copy())
            elif change=='exit':run['code']=1
            elif change=='order':s['runs'].reverse()
            else:run['peak_rss']=99
            with self.subTest(change=change),self.assertRaises(AssertionError):validate_resources(s,r)

    def test_code_refusals(self):
        text=code();self.assertEqual(len(inspect(text)['methods']),4)
        for bad in [text.replace('(FullOpts)','(Tier0)',1),text.replace('vfmadd213ps','vmulps',1),
                    text.replace('       inc rax','       call helper\n       inc rax',1),
                    text.replace('       inc rax','       vmovups zmm1, zmmword ptr [rsp+0x20]\n       inc rax',1),
                    text+text,text.replace('Blocked:PackedTile8','Wrong:PackedTile8')]:
            with self.assertRaises(AssertionError):inspect(bad)


if __name__=='__main__':unittest.main()
