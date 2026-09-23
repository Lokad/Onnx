"""Named background profiles remain fully accounted and reconciled."""
import copy
import unittest
from selected_stacks import inspect,cross_export


def fixture():
    profiles = [dict(type='evented',name='Thread (123)',unit='milliseconds',startValue=0,endValue=1,
        events=[dict(type='O',at=0,frame=0),dict(type='O',at=0,frame=1),dict(type='C',at=1,frame=1),dict(type='C',at=1,frame=0)]),
        dict(type='evented',name='Thread (456) (.NET Finalizer Thread)',unit='milliseconds',startValue=0,endValue=2,
             events=[dict(type='O',at=0,frame=2),dict(type='C',at=2,frame=2)])]
    s={'$schema':'https://www.speedscope.app/file-format-schema.json','shared':{'frames':[{'name':n} for n in ['Request','Kernel','Finalizer']]},'profiles':profiles}
    c=dict(displayTimeUnit='ms',stackFrames={str(i):r for i,r in enumerate(s['shared']['frames'])},traceEvents=[])
    for p,tid in zip(profiles,[123,456],strict=True):
        c['traceEvents'].extend(dict(cat='sampleEvent',ph='B' if r['type']=='O' else 'E',pid=0,tid=tid,sf=r['frame'],ts=r['at']*1000) for r in p['events'])
    return s,c


class NamedThreadTests(unittest.TestCase):
    def test_background_is_counted_and_cross_checked(self):
        s,c=fixture();value=inspect(s,{'request':'Request'})
        self.assertEqual(value['selected_seconds'],{'request':.001})
        self.assertEqual(value['outside_marker_seconds'],.002)
        self.assertEqual(len(value['profiles']),2)
        self.assertEqual(value['profiles'][1]['name'],'Thread (456) (.NET Finalizer Thread)')
        self.assertEqual(cross_export(s,c),dict(passed=True,profiles=2,events=6))
        c['traceEvents'].pop()
        with self.assertRaises(ValueError):cross_export(s,c)

    def test_duplicate_ids_and_invalid_names_rejected(self):
        for name in ['Thread (123) (Other)','Thread (bad)','Thread (456) ()','Thread (456) (line\nbreak)']:
            s,c=fixture();s['profiles'][1]['name']=name
            with self.assertRaises(AssertionError):inspect(s,{'request':'Request'})

    def test_original_balance_and_marker_checks_remain(self):
        s,c=fixture();s['profiles'][1]['events'][1]['frame']=1
        with self.assertRaises(AssertionError):inspect(s,{'request':'Request'})
        s,c=fixture();s['shared']['frames'][0]['name']='Unknown'
        with self.assertRaises(AssertionError):inspect(s,{'request':'Request'})


if __name__=='__main__':unittest.main()
