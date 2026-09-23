"""Synthetic accounting failures must be rejected before any trace is captured."""
import copy,unittest
from audit import reconcile,CLR,CUSTOM
def fixture():
 capture=dict(entries=[]);value=dict(passed=True,diagnosticOnly=True,protocol='parakeet-dispatch-events-v1',calls=2520,warmups=1260,measured=1260,frequency=1000,pid=7,nativeThread=9,rows=[]);events=[dict(index=0,provider=CLR,name='GC/Start',id=1,pid=7,thread=9,ms=0,payload={})]
 for i in range(21):
  capture['entries'].append(dict(name=str(i),node=i,m=51,k=1024,n=4096,y=dict(sha256='hash')))
  row=dict(index=i,name=str(i),node=i,m=51,reduction=1024,columns=4096,exact=True,guards=True,inputs=True,output='hash',preparationTicks=1,addressesBefore=dict(a=1,b=2,c=3),addressesAfter=dict(a=1,b=2,c=3),clocks=[]);value['rows'].append(row)
  for j in range(120):
   t=(i*120+j)*10+1
   row['clocks'].append(dict(iteration=j,warmup=j<60,marker=t,start=t+1,stop=t+2,ticks=1,gc0=0,gc1=0,gc2=0,after0=0,after1=0,after2=0,allocated=0,allocatedAfter=0))
   for identifier,name,counter in [(1,'Begin',t),(2,'End',t+2)]:events.append(dict(index=len(events),provider=CUSTOM,name=name,id=identifier,pid=7,thread=9,ms=counter,payload=dict(fixture=str(i),iteration=str(j),counter=str(counter))))
 summary=dict(complete=True,lost=0,clr_events=1,recorded=len(events),counts={'GC/Start':1},allCounts={CLR+':GC/Start':1,CUSTOM+':Begin':2520,CUSTOM+':End':2520})
 return value,events,summary,capture
class Accounting(unittest.TestCase):
 def setUp(self):self.args=fixture()
 def test_complete(self):self.assertEqual(reconcile(*self.args)['calls'],2520)
 def reject(self):
  with self.assertRaises(AssertionError):reconcile(*self.args)
 def test_missing(self):self.args[1].pop();self.reject()
 def test_duplicate(self):self.args[1][2]=copy.deepcopy(self.args[1][1]);self.reject()
 def test_wrong_pid(self):self.args[1][1]['pid']=8;self.reject()
 def test_wrong_thread(self):self.args[1][1]['thread']=8;self.reject()
 def test_wrong_counter(self):self.args[1][1]['payload']['counter']='2';self.reject()
 def test_wrong_order(self):self.args[1][2]['ms']=0;self.reject()
 def test_lost(self):self.args[2]['lost']=1;self.reject()
 def test_missing_accounting(self):self.args[2]['allCounts'].pop(CUSTOM+':Begin');self.reject()
 def test_clr_accounting(self):self.args[2]['counts']['GC/Start']=2;self.reject()
if __name__=='__main__':unittest.main()
