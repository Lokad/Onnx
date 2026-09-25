"""Reject misleading observer equivalence, especially changed branch/cleanup paths."""
import copy
import unittest

from il_check import PROFILER_CALL, normalize, same_except_markers


def instruction(offset, opcode, operand=''):
    return dict(offset=offset, opcode=opcode, operand=operand)


def body(instructions, exceptions=None):
    return dict(InitLocals=True, MaxStackSize=2, locals=[], exceptions=exceptions or [], instructions=instructions)


def marker(offset, stage=1001):
    return [instruction(offset, 'ldc.i4', stage.to_bytes(4, 'little', signed=True).hex()),
            instruction(offset + 5, 'call', PROFILER_CALL)]


class MarkerEquivalence(unittest.TestCase):
    def setUp(self):
        self.original = body([
            instruction(0, 'ldarg.0'), instruction(1, 'brfalse.s', '02'),
            instruction(3, 'ldc.i4.1'), instruction(4, 'pop'), instruction(5, 'ret')])
        self.observed = body([
            *marker(0), instruction(10, 'ldarg.0'), instruction(11, 'brfalse', '0c000000'),
            instruction(16, 'ldc.i4.1'), instruction(17, 'pop'), *marker(18, 1002), instruction(28, 'ret')])

    def test_markers_and_branch_width_do_not_change_behavior(self):
        result = same_except_markers(self.original, self.observed)
        self.assertEqual(result['retained_instructions'], 5)

    def test_changed_branch_target_is_rejected(self):
        changed = copy.deepcopy(self.observed)
        changed['instructions'][3]['operand'] = '01000000'  # Pop instead of return.
        with self.assertRaises(AssertionError):
            same_except_markers(self.original, changed)

    def test_changed_arithmetic_is_rejected(self):
        changed = copy.deepcopy(self.observed)
        changed['instructions'][4]['opcode'] = 'ldc.i4.2'
        with self.assertRaises(AssertionError):
            same_except_markers(self.original, changed)

    def test_unknown_stage_is_rejected(self):
        changed = copy.deepcopy(self.observed)
        changed['instructions'][0]['operand'] = (7777).to_bytes(4, 'little').hex()
        with self.assertRaises(AssertionError):
            same_except_markers(self.original, changed)

    def test_profiler_argument_cannot_hide_a_load(self):
        changed = copy.deepcopy(self.observed)
        changed['instructions'][0] = instruction(0, 'ldarg.1')
        with self.assertRaises(AssertionError):
            same_except_markers(self.original, changed)

    def test_branch_inside_removed_pair_is_rejected(self):
        changed = copy.deepcopy(self.observed)
        changed['instructions'][3]['operand'] = '07000000'  # The second marker call.
        with self.assertRaises(AssertionError):
            same_except_markers(self.original, changed)

    def test_changed_local_or_stack_is_rejected(self):
        for key, value in [('locals', [dict(type='System.Int32', IsPinned=False)]), ('MaxStackSize', 3), ('InitLocals', False)]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.observed)
                changed[key] = value
                with self.assertRaises(AssertionError):
                    same_except_markers(self.original, changed)

    def test_cleanup_region_must_still_cover_same_work(self):
        original = body([instruction(0, 'nop'), instruction(1, 'leave.s', '02'),
                         instruction(3, 'nop'), instruction(4, 'endfinally'), instruction(5, 'ret')],
                        [dict(flags=2, TryOffset=0, TryLength=3, HandlerOffset=3, HandlerLength=2, filter=-1, caught=None)])
        observed = body([*marker(0), instruction(10, 'nop'), instruction(11, 'leave.s', '0c'),
                         *marker(13, 1003), instruction(23, 'nop'), instruction(24, 'endfinally'), instruction(25, 'ret')],
                        [dict(flags=2, TryOffset=0, TryLength=13, HandlerOffset=13, HandlerLength=12, filter=-1, caught=None)])
        same_except_markers(original, observed)
        observed['exceptions'][0]['TryLength'] = 11
        with self.assertRaises(AssertionError):
            same_except_markers(original, observed)

    def test_original_stage_labels_cannot_be_removed(self):
        original = body([instruction(0, 'ldc.i4.1'), instruction(1, 'call', PROFILER_CALL), instruction(6, 'ret')])
        observed = body([*marker(0), instruction(10, 'ret')])
        with self.assertRaises(AssertionError):
            same_except_markers(original, observed)

    def test_switch_targets_are_normalized_and_checked(self):
        original = body([instruction(0, 'ldarg.0'), instruction(1, 'switch', '0000000002000000'),
                         instruction(14, 'ldc.i4.1'), instruction(15, 'pop'), instruction(16, 'ret')])
        observed = body([*marker(0), instruction(10, 'ldarg.0'), instruction(11, 'switch', '000000000c000000'),
                         instruction(24, 'ldc.i4.1'), instruction(25, 'pop'), *marker(26), instruction(36, 'ret')])
        same_except_markers(original, observed)
        observed['instructions'][3]['operand'] = '010000000c000000'
        with self.assertRaises(AssertionError):
            same_except_markers(original, observed)

    def test_duplicate_or_mid_instruction_targets_are_rejected(self):
        changed = copy.deepcopy(self.observed)
        changed['instructions'][3]['operand'] = '08000000'
        with self.assertRaises(AssertionError):
            normalize(changed)
        changed['instructions'][1]['offset'] = 0
        with self.assertRaises(AssertionError):
            normalize(changed)


if __name__ == '__main__':
    unittest.main()
