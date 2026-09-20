"""Language dispatch and corrupted real-selection checks before inference."""
from pathlib import Path
import copy
import types
import unittest
# Load the frontend before Arrow's Windows DLLs, as in the qualified audio lanes.
from whisper_adapter import Whisper
import numpy as np
from audit_inputs import check_selection
from common import read


class EvidenceTests(unittest.TestCase):
    def test_actual_native_decoder_receives_each_declared_language(self):
        model=Whisper.__new__(Whisper)
        model.generation=dict(eos_token_id=5,decoder_start_token_id=10,lang_to_id={f'<|{v}|>':i+11 for i,v in enumerate(('en','fr','de','es','it'))},
                              task_to_id=dict(transcribe=16),no_timestamps_token_id=17,suppress_tokens=[],begin_suppress_tokens=[])
        model.tokenizer=types.SimpleNamespace(token_to_id=lambda v:8,decode=lambda ids,skip_special_tokens:'')
        model.features=lambda pcm:np.zeros((1,128,3000),np.float32)
        captured=[]
        def graph(name,feeds):
            if name=='encoder':return {'last_hidden_state':np.zeros((1,1,1),np.float32)}
            self.assertEqual(name,'first');captured.append(feeds['input_ids'].tolist()[0])
            logits=np.zeros((1,4,20),np.float32);logits[:,:,-15]=10
            return dict(logits=logits)
        model.graph=graph
        for language in ('en','fr','de','es','it'):
            result=model(np.ones(16,np.float32),language)
            self.assertEqual(result['token_ids'],[5]);self.assertEqual(result['stop_reason'],'EndToken')
        self.assertEqual(captured,[[10,i,16,17] for i in range(11,16)])
        with self.assertRaises(AssertionError):model(np.ones(16,np.float32),'unknown')

    def test_real_selection_refuses_changed_labels_coverage_and_provenance(self):
        base=Path(__file__).resolve().parents[3]/'artifacts/asr-multilingual-v2-20260920/inputs'
        if not base.exists():self.skipTest('Optional prepared multilingual evidence is absent')
        selection=read(base/'selection.json');audio=read(base/'audio.json')
        inventories=[v['inventory'] for v in selection['locales']]
        self.assertEqual(len(check_selection(selection,audio,inventories)),40)
        mutations=[lambda a:a['cases'].pop(),lambda a:a['cases'].__setitem__(1,a['cases'][0]),
            lambda a:a['cases'][0].__setitem__('language','fr'),lambda a:a['cases'][0].__setitem__('reference_text','altered label'),
            lambda a:a['cases'][0].__setitem__('source_id',-1),lambda a:a['cases'][0].__setitem__('samples',1),
            lambda a:a['cases'][0].__setitem__('condition','noise10db'),lambda a:a.__setitem__('protocol','unknown')]
        for mutation in mutations:
            damaged=copy.deepcopy(audio);mutation(damaged)
            with self.assertRaises(AssertionError):check_selection(selection,damaged,inventories)
        damaged=copy.deepcopy(selection);damaged['locales'][0]['selected'].reverse()
        with self.assertRaises(AssertionError):check_selection(damaged,audio,inventories)


if __name__=='__main__':unittest.main()
