"""Reject corrupt real benchmark records before starting the Linux comparison."""
from pathlib import Path
import copy, hashlib, sys, unittest
import numpy as np
from protocol import read,check_result,validate_records,schedule,check_sample,LIMITS

ROOT=Path(__file__).resolve().parents[3]


class ProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples={}
        for family,artifact,worker in [('parakeet','audio-ort-baseline-v2-20260919','01-parakeet-managed'),
                                       ('pyannote','audio-ort-baseline-v2-20260919','03-pyannote-managed'),
                                       ('whisper','whisper-ort-baseline-20260919','01-whisper-managed')]:
            base=ROOT/'artifacts'/artifact;manifest=read(base/'inputs'/(family+'.json'))
            for case in manifest['cases']:
                pcm=np.load(ROOT/case['pcm']['path'],allow_pickle=False);case['raw_sha256']=hashlib.sha256(pcm.tobytes()).hexdigest()
            value=read(base/'conformance'/worker/'result.json');validate_records(value,manifest,'conformance')
            cls.examples[family]=(value,manifest)

    def test_complete_original_records(self):
        for value,manifest in self.examples.values():validate_records(value,manifest,'conformance')
        self.assertEqual(len(schedule('conformance')),6);self.assertEqual(len(schedule('timing')),12)
        self.assertEqual(schedule('timing')[:4],[('parakeet',e) for e in ['managed','ort','ort','managed']])

    def test_complete_original_timing_records(self):
        processes=calls=0
        for artifact in ['audio-ort-baseline-v2-20260919','whisper-ort-baseline-20260919']:
            for path in (ROOT/'artifacts'/artifact/'timing').glob('*/result.json'):
                value=read(path)
                # The old native Whisper record lacks the new cross-host feature summary.
                if value['family']=='whisper' and value['engine']=='ort':continue
                validate_records(value,self.examples[value['family']][1],'timing')
                processes+=1;calls+=len(value['records'])
                damaged=copy.deepcopy(value);damaged['records'][0],damaged['records'][1]=damaged['records'][1],damaged['records'][0]
                with self.assertRaises(AssertionError):validate_records(damaged,self.examples[value['family']][1],'timing')
        self.assertEqual((processes,calls),(10,544))

    def test_corrupted_real_records(self):
        for family,(original,manifest) in self.examples.items():
            changes=[lambda v:v['records'].pop(),lambda v:v['records'].append(copy.deepcopy(v['records'][0])),
                     lambda v:v['records'][0].update(seconds=-1),lambda v:v['records'][0].update(frequency=0),
                     lambda v:v['records'][0].update(start_ticks=v['records'][0]['end_ticks']),
                     lambda v:v['records'][0].update(ownership=False),lambda v:v['records'][0].update(input_sha256='0'*64),
                     lambda v:v['records'][0].update(phase='measured'),lambda v:v.update(affinity=1),
                     lambda v:v.update(held_outputs_unchanged=False),lambda v:v.update(held_outputs_unchanged=1),
                     lambda v:v['records'][0].update(ownership=1),lambda v:v.update(conformance=1),
                     lambda v:v.update(flags={'DOTNET_TieredCompilation':'0'})]
            for change in changes:
                value=copy.deepcopy(original);change(value)
                with self.assertRaises(AssertionError,msg=family):validate_records(value,manifest,'conformance')

    def test_decisions_centroids_and_endpoints(self):
        for family in ['parakeet','whisper']:
            value,manifest=self.examples[family];actual=copy.deepcopy(value['records'][0]['result']);expected=manifest['cases'][0]['expected']
            actual['token_ids'][0]=float(actual['token_ids'][0])
            with self.assertRaises(AssertionError):check_result(actual,expected,family=family)
            actual['token_ids'][0]=True
            with self.assertRaises(AssertionError):check_result(actual,expected,family=family)
        value,manifest=self.examples['pyannote'];expected=manifest['cases'][0]['expected']
        for category in ['centroid','endpoint','speaker']:
            actual=copy.deepcopy(value['records'][0]['result'])
            if category=='centroid':actual['speakers'][0]['centroid'][0]+=0.01
            elif category=='endpoint':actual['intervals'][0][0]+=1e-6
            else:actual['speakers'][0]['speaker']+=1
            with self.assertRaises(AssertionError):check_result(actual,expected,family='pyannote')

    def test_resource_refusals(self):
        sample=dict(seconds=1.,available=2*1024**3,disk=64*1024**2,members=[dict(pid=1,birth=1,rss=1024,affinity=[2],threads=[dict(tid=1,affinity=[2])])])
        check_sample(sample)
        for key,value in [('seconds',LIMITS['seconds']),('available',0),('disk',0),('members',[])]:
            row=copy.deepcopy(sample);row[key]=value
            with self.assertRaises(AssertionError):check_sample(row)
        for key,value in [('rss',LIMITS['rss']),('affinity',[0])]:
            row=copy.deepcopy(sample);row['members'][0][key]=value
            with self.assertRaises(AssertionError):check_sample(row)
        row=copy.deepcopy(sample);row['members'][0]['threads'][0]['affinity']=[0]
        with self.assertRaises(AssertionError):check_sample(row)


if __name__=='__main__':unittest.main()
