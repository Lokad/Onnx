"""Exercise the graph schema on prior retained evidence, never as new-product scores."""
import unittest
from protocol import read
from prepare import ROOT
from graph_prerequisite import validate


class GraphPrerequisite(unittest.TestCase):
    def fixture(self):
        graph=ROOT/'artifacts/parakeet-owned-batch-isolation-graphs-amd-20260925'
        short=ROOT/'artifacts/e5-steady-short-amd-20260925'
        value=read(graph/'analysis.json');replacement=read(short/'analysis.json')
        value['performance'][0]=replacement['performance']
        value['short_consumer']=replacement['consumer']
        value['products']=read(graph/'payload.json')['products']
        value['clocks']=value['clocks']-4689+replacement['clocks']
        jobs=read(graph/'payload.json')['jobs'];value['setups']=[]
        for name in jobs:
            folder=short if '-e5-8tok-' in name else graph
            result=read(folder/'collected'/name/'output/result.json')
            value['setups'].append(dict(process=name,seconds=result['setup_seconds']))
        proof=dict(passed=True,admitted=True,all_controls_passed=True)
        return value,proof,value['products']

    def test_retained_corrected_protocol_schema(self):validate(*self.fixture())

    def test_failed_control(self):
        value,proof,products=self.fixture();value['performance'][0]['controls'][0]['passed']=False
        with self.assertRaises(AssertionError):validate(value,proof,products)

    def test_missing_case(self):
        value,proof,products=self.fixture();value['performance'].pop()
        with self.assertRaises(AssertionError):validate(value,proof,products)

    def test_wrong_product(self):
        value,proof,products=self.fixture()
        with self.assertRaises(AssertionError):validate(value,proof,{'current':products['candidate'],'candidate':products['current']})

    def test_wrong_short_consumer(self):
        value,proof,products=self.fixture();value['short_consumer']['consumer']=value['consumer']['consumer']
        with self.assertRaises(AssertionError):validate(value,proof,products)

    def test_missing_clocks(self):
        value,proof,products=self.fixture();value['clocks']-=1
        with self.assertRaises(AssertionError):validate(value,proof,products)


if __name__=='__main__':unittest.main()
