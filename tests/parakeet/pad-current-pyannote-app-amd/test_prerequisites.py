"""Reject incompatible products, missing equality and failed admission."""
import unittest
from protocol import read
from prepare import PRIOR
from prerequisites import validate


class Prerequisites(unittest.TestCase):
    def fixture(self):
        reports={key:read(folder/'analysis.json') for key,folder in PRIOR.items()}
        return reports,dict(identities=reports['models']['identities'],consumers=reports['baseline']['consumers'])

    def test_actual_closed_pair(self):validate(*self.fixture())

    def test_wrong_data_identity(self):
        reports,spec=self.fixture();spec['identities']['candidate']['Lokad.Onnx.Data.dll']['sha256']='0'*64
        with self.assertRaises(AssertionError):validate(reports,spec)

    def test_missing_native_equality(self):
        reports,spec=self.fixture();reports['parakeet']['results']['candidate-native-512']['native']['exact_selected_comparisons'].pop()
        with self.assertRaises(AssertionError):validate(reports,spec)

    def test_changed_transcription(self):
        reports,spec=self.fixture();reports['parakeet']['results']['candidate-public-256']['complete_selected_results_exact']=False
        with self.assertRaises(AssertionError):validate(reports,spec)

    def test_changed_pyannote_result(self):
        reports,spec=self.fixture();reports['models']['results']['candidate']['complete_public_results_exact']=False
        with self.assertRaises(AssertionError):validate(reports,spec)

    def test_different_meeting_consumer(self):
        reports,spec=self.fixture();spec['consumers']['NaturalMeetings']['sha256']='0'*64
        with self.assertRaises(AssertionError):validate(reports,spec)

    def test_failed_parakeet_admission(self):
        reports,spec=self.fixture();reports['parakeet-app']['performance']['admitted']=False
        with self.assertRaises(AssertionError):validate(reports,spec)


if __name__=='__main__':unittest.main()
