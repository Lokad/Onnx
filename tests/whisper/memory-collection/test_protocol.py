"""Reject missing interventions, changed ownership, invalid clocks and false GC evidence."""
import copy,unittest
from memory_protocol import INTEGER_FIELDS,validate_memory


def fixture():
    records=[];collections=[];generation=4
    def snap(t):
        d={k:100 for k in INTEGER_FIELDS};d.update(ticks=t,allocated_total=t,last_gc_index=generation,last_gc_generation=2,
            collections=[generation+10,generation+2,generation],last_gc_concurrent=False,last_gc_compacted=True)
        return d
    for i in range(1,21):
        t=i*100;before=snap(t);after=snap(t+20)
        records.append(dict(start_ticks=t+5,end_ticks=t+15,gc_before=before['collections'],gc_after=after['collections'],memory_before=before,memory_after=after))
        if i in [8,16,20]:
            before=snap(t+30);generation+=1;after=snap(t+60)
            collections.append(dict(after_call=i,before=before,after=after,start_ticks=t+40,end_ticks=t+50,frequency=100,seconds=.1,held_outputs_unchanged=True,inputs_unchanged=True))
    return dict(diagnostic='explicit-gen2-after-8-16-20',records=records,collections=collections)


class Checks(unittest.TestCase):
    def test_complete_without_assuming_reclamation(self):
        value=fixture();value['collections'][0]['after']['managed_estimate']=1000;validate_memory(value)

    def test_corrupt_evidence(self):
        changes=[lambda v:v['collections'].pop(),lambda v:v['collections'].reverse(),
            lambda v:v['collections'][0].update(after_call=7),lambda v:v['records'][1]['memory_before'].update(ticks=1),
            lambda v:v['records'][0]['memory_after'].update(allocated_total=0),
            lambda v:v['collections'][0]['after'].update(collections=v['collections'][0]['before']['collections']),
            lambda v:v['collections'][0]['after'].update(last_gc_index=v['collections'][0]['before']['last_gc_index']),
            lambda v:v['collections'][0].update(held_outputs_unchanged=False),lambda v:v['collections'][0].update(inputs_unchanged=False),
            lambda v:v['collections'][0].update(seconds=.2),lambda v:v['records'][0]['memory_after'].update(rss=float('nan'))]
        for change in changes:
            with self.subTest(change=change):
                value=copy.deepcopy(fixture());change(value)
                with self.assertRaises((AssertionError,KeyError,TypeError,ValueError)):validate_memory(value)


if __name__=='__main__':unittest.main()
