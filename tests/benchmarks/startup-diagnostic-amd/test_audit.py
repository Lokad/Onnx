import base64,copy,struct,unittest
from audit import CUSTOM,CLR,reconcile

def fixture():
    clocks=[];events=[dict(index=0,provider=CLR,name='test',id=1,pid=7,thread=8,ms=0,rawBase64='',rawLength=0)]
    for i in range(1200):
        counter=100*i+1
        clocks.append(dict(index=i,warmup=i<60,marker=counter,start=counter+1,end=counter+11,ticks=10,frequency=1000,
            cpuBefore=i,cpuAfter=i+1,allocated=i,allocatedAfter=i+1,gc0=0,gc1=0,gc2=0,after0=0,after1=0,after2=0))
        for event_id,count in [(1,counter),(2,counter+11)]:
            raw=struct.pack('<iiq',0,i,count)
            events.append(dict(index=len(events),provider=CUSTOM,name=str(event_id),id=event_id,pid=7,thread=8,ms=count,
                rawBase64=base64.b64encode(raw).decode(),rawLength=16,payload=dict(fixture='0',iteration=str(i),counter=str(count))))
    return dict(passed=True,diagnosticOnly=True,mode='diagnostic',calls=1200,clocks=clocks,pid=7,nativeThread=8),events,dict(
        complete=True,lost=0,clr_events=1,protocol='all-event-records-v1',recorded=len(events),allCounts={CLR+':test':1,CUSTOM+':1':1200,CUSTOM+':2':1200})

class AuditTests(unittest.TestCase):
    def test_all_calls_and_fixed_blocks(self):
        value,events,summary=fixture();r=reconcile(value,events,summary)
        self.assertEqual((len(r['calls']),len(r['blocks']),r['markers']),(1200,20,2400))
        self.assertEqual(r['blocks'][-1]['last'],1199)
    def test_loss_and_missing_marker_rejected(self):
        value,events,summary=fixture();summary['lost']=1
        with self.assertRaises(AssertionError):reconcile(value,events,summary)
        summary['lost']=0
        with self.assertRaises(AssertionError):reconcile(value,events[:-1],summary)
    def test_wrong_thread_or_counter_rejected(self):
        value,events,summary=fixture();events[12]['thread']=9
        with self.assertRaises(AssertionError):reconcile(value,events,summary)
        events[12]['thread']=8;events[12]['rawBase64']=base64.b64encode(struct.pack('<iiq',0,5,999)).decode()
        with self.assertRaises(AssertionError):reconcile(value,events,summary)
    def test_clock_and_collection_regression_rejected(self):
        value,events,summary=fixture();value['clocks'][70]['ticks']=11
        with self.assertRaises(AssertionError):reconcile(value,events,summary)
        value['clocks'][70]['ticks']=10;value['clocks'][70]['after2']=-1
        with self.assertRaises(AssertionError):reconcile(value,events,summary)

if __name__=='__main__':unittest.main()
