import unittest
from prepare_labels import parse_words,reference,union


def xml(body):
    return ('<?xml version="1.0" encoding="ISO-8859-1"?><nite:root xmlns:nite="http://nite.sourceforge.net/" nite:id="ES2004a.A.words">'+body+'</nite:root>').encode('latin-1')


class LabelTests(unittest.TestCase):
    def test_encoding_punctuation_attributes_and_boundary(self):
        words,counts=parse_words(xml('<w nite:id="ES2004a.A.words0" starttime="0" endtime="1">café</w><w nite:id="ES2004a.A.words1" starttime="1" endtime="1" punc="true">.</w><w nite:id="ES2004a.A.words2" starttime="1" endtime="3" trunc="true">s</w>'),'ES2004a','A')
        selected=reference(words,2)
        self.assertEqual(selected['text'],'café');self.assertEqual(counts,{'w':3})
        self.assertEqual(selected['excluded_boundary_words'][0]['text'],'s')
        self.assertEqual(words[1]['attributes']['trunc'],'true')
        self.assertEqual(union(words,{'A':'person'},2),[[0.,2.,'person']])

    def test_order_and_exact_end(self):
        words=[dict(start='1',end='2',speaker='B',index=0,text='second'),dict(start='1',end='2',speaker='A',index=1,text='first')]
        self.assertEqual(reference(words,2)['text'],'first second')
        self.assertEqual(reference(words,1)['text'],'')

    def test_invalid_metadata_refused(self):
        valid='<w nite:id="ES2004a.A.words0" starttime="0" endtime="1">hi</w>'
        for body in [valid+valid,valid.replace('words0','wrong'),valid.replace('starttime="0"','starttime="NaN"'),
                     valid.replace('starttime="0"','starttime="-1"'),valid.replace('endtime="1"','endtime="0"'),valid.replace('>hi<','><')]:
            with self.assertRaises(ValueError):parse_words(xml(body),'ES2004a','A')
        with self.assertRaises(ValueError):parse_words(xml(valid),'ES2004a','B')


if __name__=='__main__':unittest.main()
