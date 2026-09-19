import copy,unittest
import numpy as np
from audit import validate,canonical_refusals

class Refusals(unittest.TestCase):
    def test_infinity_labels_preserve_sign_and_other_refusals(self):
        self.assertEqual(['nonfinite-Infinity','nonfinite--Infinity','nonfinite-NaN','missing'],canonical_refusals(['nonfinite-∞','nonfinite--∞','nonfinite-NaN','missing']))
    def setUp(self):
        self.pcm=np.full(496000,.1,np.float32);self.case=dict(max_tokens=3,max_tokens_per_frame=10,max_windows=1)
        self.result=dict(text='A',duration_seconds=31,processed_seconds=30,stop_reason='WindowLimit',windows=[dict(start_seconds=0,audio_seconds=30,boundary='HardLimit',
            decoding=dict(text='A',token_ids=[1],frame_indices=[0],duration_frames=[1],stop_reason='EndOfAudio',encoded_frames=376,decoder_calls=376))])
    def check(self,r):validate(r,self.case,self.pcm,{1:'A',2:'B'})
    def test_valid(self):self.check(self.result)
    def test_corruptions(self):
        changes=[('duration_seconds',32),('processed_seconds',31),('text','B'),('stop_reason','Completed'),('windows',[])]
        for key,value in changes:
            with self.subTest(key=key):
                r=copy.deepcopy(self.result);r[key]=value
                with self.assertRaises(ValueError):self.check(r)
    def test_window_corruptions(self):
        for key,value in [('start_seconds',1),('audio_seconds',29),('boundary','Quiet')]:
            with self.subTest(key=key):
                r=copy.deepcopy(self.result);r['windows'][0][key]=value
                with self.assertRaises(ValueError):self.check(r)
    def test_decoding_corruptions(self):
        for key,value in [('text','B'),('token_ids',[2]),('frame_indices',[376]),('duration_frames',[5]),('encoded_frames',375),('decoder_calls',10000),('stop_reason','SilentInput')]:
            with self.subTest(key=key):
                r=copy.deepcopy(self.result);r['windows'][0]['decoding'][key]=value
                with self.assertRaises(ValueError):self.check(r)
    def test_partial_cannot_commit_window(self):
        r=copy.deepcopy(self.result);r['windows'][0]['decoding'].update(token_ids=[1,1,1],frame_indices=[0,1,2],duration_frames=[1,1,1],text='AAA',stop_reason='TokenLimit')
        r.update(stop_reason='TokenLimit',text='AAA')
        with self.assertRaises(ValueError):self.check(r)

if __name__=='__main__':unittest.main()
