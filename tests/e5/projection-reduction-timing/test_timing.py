import copy, itertools, unittest
from common import *


def bank_fixture(worker=0,index=0):
    sched=schedule();definition=BANKS[index];clock=1
    def samples(cycles,first=False):
        nonlocal clock
        out=[]
        for cycle in range(cycles):
            for position,mode in enumerate(range(4) if first else sched['cycles'][worker][index][cycle%48]):
                duration=200 if first or cycles==16 else 100 if mode<2 else 95 if mode==2 else 96
                out.append(dict(cycle=-1 if first else cycle,position=position,mode=mode,start=clock,end=clock+duration,allocated_thread=0,allocated_total=0,gc0=0,gc1=0,gc2=0));clock+=duration+1
        return out
    hashes=dict(input384='a'*64,input1536='b'*64,packed=[f'{i:064x}' for i in range(72)],outputs=[[f'{i:064x}' for i in range(72)] for _ in range(4)])
    return dict(**definition,worker=worker,bank_index=index,before=hashes,after=copy.deepcopy(hashes),first=samples(1,True),conditioning=samples(16),conditioning_cycles=16,conditioning_ticks=[3200]*4,measured=samples(48))


class Tests(unittest.TestCase):
    def test_schedule_exact_balance(self):
        sched=schedule()
        for worker in range(4):
            self.assertEqual(sorted(sched['orders'][worker]),list(range(5)))
            for bank in sched['cycles'][worker]:
                for block in [bank[:24],bank[24:]]:self.assertEqual(sorted(map(tuple,block)),list(itertools.permutations(range(4))))

    def test_coverage_and_damaged_records(self):
        valid=bank_fixture();self.assertEqual(validate_bank(valid,0,0,1000,schedule())['measured_calls'],192)
        mutations=[lambda b:b['measured'].pop(),lambda b:b['measured'].append(b['measured'][0]),
            lambda b:b['measured'][0].update(mode=5),lambda b:b['measured'][0].update(cycle=5),
            lambda b:b['measured'][0].update(end=b['measured'][0]['start']),lambda b:b['measured'][0].update(start=1.0),
            lambda b:b['measured'][0].update(allocated_thread=1,allocated_total=1),lambda b:b['measured'][0].update(gc0=1),
            lambda b:b['after']['outputs'][2].__setitem__(0,'f'*64),lambda b:b.update(conditioning_ticks=[3199]*4),
            lambda b:b.update(conditioning_cycles=15),lambda b:b.update(worker=2)]
        for mutation in mutations:
            bad=copy.deepcopy(valid);mutation(bad)
            with self.assertRaises(AssertionError):validate_bank(bad,0,0,1000,schedule())

    def test_prospective_verdict_and_failures(self):
        workers=[]
        for worker in range(4):
            raw=[bank_fixture(worker,index) for index in range(5)]
            workers.append(dict(identity=dict(frequency=1000),raw=raw,banks=[validate_bank(b,worker,i,1000,schedule()) for i,b in enumerate(raw)]))
        result=verdict(workers);self.assertEqual(result['verdict'],'pass');self.assertEqual(result['nominated'],2)
        bad=copy.deepcopy(workers);bad[1]['banks'][2]['means_ms'][1]*=1.1
        self.assertEqual(verdict(bad)['verdict'],'inconclusive')
        bad=copy.deepcopy(workers)
        for w in bad:
            for bank in w['banks']:bank['means_ms'][2:]=[101,102]
        self.assertEqual(verdict(bad)['verdict'],'rejected')
        bad=copy.deepcopy(workers)
        for w in bad:
            for bank in w['banks']:bank['means_ms'][3]=bank['means_ms'][2]
        self.assertEqual(verdict(bad)['nominated'],3)

    def test_resources_and_faults(self):
        identity=dict(complete=True,code=0,limits=LIMITS,started=0,ended=5,supervisor=dict(pid=1,start=1,affinity='0'),runs=[]);samples={}
        for i in range(4):
            pid=i+2;name=f'worker{i}';member=dict(pid=pid,start=pid,affinity='2',group=pid,rss=100,cpu_seconds=.1)
            identity['runs'].append(dict(name=name,pid=pid,start=pid,started=i+.01,ended=i+.4,code=0,seconds=.3,flags={},samples=2,
                peak_rss=100,preflight_available=5*1024**3,preflight_disk=200*1024**2,members={str(pid):pid},foreign_activity=[]))
            samples[name]=[dict(seconds=.1*j,members=[member],available_memory=2*1024**3,cpu={k:[j]*10 for k in ['cpu','cpu2']}) for j in [1,2]]
        self.assertEqual(len(validate_resources(identity,samples)['births']),5)
        for field,value in [('available_memory',0),('seconds',601),('members',[dict(member,affinity='0')]),('cpu',{'cpu':[1]*10})]:
            bad=copy.deepcopy(samples);bad['worker0'][0][field]=value
            with self.assertRaises(AssertionError):validate_resources(identity,bad)
        for field,value in [('complete',False),('code',1),('limits',{})]:
            bad=copy.deepcopy(identity);bad[field]=value
            with self.assertRaises(AssertionError):validate_resources(bad,samples)


if __name__=='__main__':unittest.main()
