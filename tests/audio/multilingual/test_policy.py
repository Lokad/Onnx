import unittest
import numpy as np
from common import select, normalize, variants


class PolicyTests(unittest.TestCase):
    def test_selection_is_order_independent_and_excludes_repeated_sentence(self):
        rows = [dict(id=i, filename=str(i)+'.wav', gender=g, samples=s*16000)
                for i,g,s in [(1,0,5),(2,0,15),(3,1,5),(4,1,15),(5,0,7),(6,1,14)]]
        rows.append(dict(id=1,filename='other-speaker.wav',gender=1,samples=5*16000))
        self.assertEqual([1,2,3,4], [r['id'] for r in select(list(reversed(rows)))])
        with self.assertRaises(ValueError): select(rows+[rows[0]])
        with self.assertRaises(ValueError): select([r for r in rows if not (r['gender']==1 and r['samples']>=12*16000)])

    def test_single_available_gender_keeps_four_distinct_sentences(self):
        rows=[dict(id=i,filename=str(i)+'.wav',gender=0,samples=s*16000) for i,s in [(1,5),(2,15),(3,7),(4,14)]]
        self.assertEqual([1,2,3,4],[r['id'] for r in select(rows)])
        with self.assertRaises(ValueError):select([])

    def test_unicode_policy_preserves_accents_and_embedded_apostrophes(self):
        self.assertEqual("l'accident aujourd'hui straße 42".casefold(), normalize('L’accident, AUJOURDʼHUI; Straße 42!'))
        self.assertEqual('café déjà vu', normalize('CAFÉ — déjà\tvu.'))
        self.assertEqual('quoted phrase', normalize("'quoted' \"phrase\""))
        self.assertNotEqual(normalize('café'), normalize('cafe'))
        self.assertNotEqual(normalize('42'), normalize('forty two'))

    def test_noise_has_declared_power_and_common_gain_without_clipping(self):
        pcm=np.sin(np.arange(32000,dtype=np.float64)*.05).astype(np.float32)
        a,b,info=variants(pcm,'fr_fr','fixture.wav')
        self.assertLess(info['gain'],1)
        self.assertTrue(np.array_equal(a,(pcm.astype(np.float64)*info['gain']).astype(np.float32)))
        noise=b.astype(np.float64)-a
        self.assertAlmostEqual(10*np.log10(np.dot(a.astype(np.float64),a)/np.dot(noise,noise)),10,places=5)
        again=variants(pcm,'fr_fr','fixture.wav')
        self.assertEqual(a.tobytes(),again[0].tobytes());self.assertEqual(b.tobytes(),again[1].tobytes())
        self.assertNotEqual(b.tobytes(),variants(pcm,'en_us','fixture.wav')[1].tobytes())
        for invalid in (np.zeros(10,np.float32),np.array([float('nan')],np.float32),np.array([1.1],np.float32),np.array([.5],np.float32)):
            with self.assertRaises(ValueError):variants(invalid,'fr_fr','fixture.wav')


if __name__=='__main__':unittest.main()
