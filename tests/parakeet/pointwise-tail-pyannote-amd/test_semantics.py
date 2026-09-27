import copy,unittest
from checks import semantic_agreement

class PublicSemantics(unittest.TestCase):
    def setUp(self):
        self.value=dict(Intervals=[dict(Start=0.,End=1.,Speaker=0)],ExclusiveIntervals=[dict(Start=0.,End=1.,Speaker=0)],
            Speakers=[dict(Speaker=0,Centroid=[.1]*256,HasEmbedding=True)],Status=0,AudioDuration=1.,Windows=1)
    def test_identical(self):self.assertTrue(semantic_agreement(self.value,copy.deepcopy(self.value)))
    def test_centroids_are_separately_numeric(self):
        other=copy.deepcopy(self.value);other['Speakers'][0]['Centroid'][0]+=.00000001
        self.assertTrue(semantic_agreement(self.value,other))
    def test_every_semantic_field_remains_exact(self):
        for key,changed in [('Status',1),('Windows',2),('AudioDuration',2.),('Intervals',[]),('ExclusiveIntervals',[])]:
            with self.subTest(key=key):
                other=copy.deepcopy(self.value);other[key]=changed
                with self.assertRaises(AssertionError):semantic_agreement(self.value,other)
    def test_speaker_and_embedding_state(self):
        for key,changed in [('Speaker',1),('HasEmbedding',False)]:
            other=copy.deepcopy(self.value);other['Speakers'][0][key]=changed
            with self.assertRaises(AssertionError):semantic_agreement(self.value,other)
    def test_timeline_last_bit(self):
        other=copy.deepcopy(self.value);other['Intervals'][0]['End']=1.0000000000000002
        with self.assertRaises(AssertionError):semantic_agreement(self.value,other)
    def test_unknown_top_field(self):
        other=copy.deepcopy(self.value);other['NewSemanticField']=3
        with self.assertRaises(AssertionError):semantic_agreement(self.value,other)
    def test_unknown_speaker_field(self):
        other=copy.deepcopy(self.value);other['Speakers'][0]['Confidence']=.5
        with self.assertRaises(AssertionError):semantic_agreement(self.value,other)

if __name__=='__main__':unittest.main()
