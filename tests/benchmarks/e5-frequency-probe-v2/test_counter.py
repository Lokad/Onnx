import copy
import unittest
from counter import EVENTS, epoch, intervals


def sample(stamp='1.000000000'):
    return '\n'.join(f'{stamp};123;;{name};1000000000;100.00;;' for name in EVENTS)+'\n'


class Contracts(unittest.TestCase):
    def test_complete(self):
        rows = intervals(sample()+sample('2.000000000'))
        self.assertEqual([r['elapsed_ns'] for r in rows], [1000000000, 2000000000])
        self.assertEqual(list(rows[0]['events']), EVENTS)

    def test_missing(self):
        with self.assertRaises(AssertionError): intervals('\n'.join(sample().splitlines()[:-1]))

    def test_duplicate(self):
        with self.assertRaises(AssertionError): intervals(sample()+sample().splitlines()[0]+'\n')

    def test_unsupported(self):
        with self.assertRaises(Exception): intervals(sample().replace('123;', '<not supported>;', 1))

    def test_bad_value(self):
        for value in ['NaN', 'Infinity', '-1']:
            with self.subTest(value=value), self.assertRaises(AssertionError):
                intervals(sample().replace('123;', value+';', 1))

    def test_nonmonotonic(self):
        with self.assertRaises(AssertionError): intervals(sample('2.0')+sample('1.0'))

    def test_fraction(self):
        with self.assertRaises(AssertionError): intervals(sample().replace('100.00', '101.00'))

    def test_epoch(self):
        anchors = [dict(command='start', before_ns=1000, ack_ns=1020, after_ns=1020, elapsed_ns=0),
                   dict(command='stop', before_ns=2015, ack_ns=2020, after_ns=2025, elapsed_ns=1010)]
        self.assertEqual(epoch(anchors), dict(lower_ns=1005, upper_ns=1015, uncertainty_ns=10))
        broken = copy.deepcopy(anchors); broken[1]['elapsed_ns'] += 100
        with self.assertRaises(AssertionError): epoch(broken)
        broken = copy.deepcopy(anchors); broken[1]['after_ns'] += 21000000
        with self.assertRaises(AssertionError): epoch(broken)


if __name__ == '__main__': unittest.main()
