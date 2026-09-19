import copy
import unittest
from audit import validate_recording

class Tokenizer:
    def decode(self,tokens,skip_special_tokens):
        return ''.join(chr(t+33) for t in tokens if t<50257)

class AuditRefusals(unittest.TestCase):
    def setUp(self):
        self.case=dict(samples=480000,max_windows=256,max_new_tokens=444)
        self.result=dict(text='A',duration_seconds=30,processed_seconds=30,stop_reason='Completed',
            segments=[dict(start_seconds=0,end_seconds=20,text='A',token_ids=[32])],
            windows=[dict(start_seconds=0,audio_seconds=30,advanced_seconds=30,decoding=dict(text='A',token_ids=[50365,32,51365,50257],
                stop_reason='EndToken',skipped_as_no_speech=False,no_speech_probability=.01,average_log_probability=-.1))])
    def check(self,result):validate_recording(result,self.case,Tokenizer())
    def test_valid_closed_segment(self):self.check(self.result)
    def test_incomplete_cannot_claim_completed(self):
        result=copy.deepcopy(self.result);result['processed_seconds']=20;result['windows'][0]['advanced_seconds']=20
        with self.assertRaises(ValueError):self.check(result)
    def test_uncommitted_segment_is_rejected(self):
        result=copy.deepcopy(self.result);result['segments'][0]['end_seconds']=31
        with self.assertRaises(ValueError):self.check(result)
    def test_window_gap_is_rejected(self):
        result=copy.deepcopy(self.result);result['windows'][0]['start_seconds']=.02
        with self.assertRaises(ValueError):self.check(result)
    def test_changed_aggregate_is_rejected(self):
        result=copy.deepcopy(self.result);result['text']='B'
        with self.assertRaises(ValueError):self.check(result)
    def test_control_token_is_rejected(self):
        result=copy.deepcopy(self.result);result['windows'][0]['decoding']['token_ids'][1]=50360
        with self.assertRaises(ValueError):self.check(result)
    def test_early_eos_is_rejected(self):
        result=copy.deepcopy(self.result);result['windows'][0]['decoding']['token_ids'][1]=50257
        with self.assertRaises(ValueError):self.check(result)
    def test_nonfinite_confidence_is_rejected(self):
        result=copy.deepcopy(self.result);result['windows'][0]['decoding']['average_log_probability']=float('nan')
        with self.assertRaises(ValueError):self.check(result)
    def test_false_window_limit_is_rejected(self):
        result=copy.deepcopy(self.result);result['stop_reason']='WindowLimit'
        with self.assertRaises(ValueError):self.check(result)
    def test_wrong_no_speech_decision_is_rejected(self):
        result=copy.deepcopy(self.result);result['windows'][0]['decoding']['no_speech_probability']=.99
        result['windows'][0]['decoding']['average_log_probability']=-2
        with self.assertRaises(ValueError):self.check(result)
    def test_dropped_committed_segment_is_rejected(self):
        result=copy.deepcopy(self.result);result['segments']=[];result['text']=''
        with self.assertRaises(ValueError):self.check(result)
    def test_shifted_segment_within_duration_is_rejected(self):
        result=copy.deepcopy(self.result);result['segments'][0]['start_seconds']=1
        with self.assertRaises(ValueError):self.check(result)

if __name__=='__main__':unittest.main()
