import unittest
from unittest.mock import patch

import ci_scope


class ScopeTests(unittest.TestCase):
    def test_prose_only_uses_lightweight_job(self):
        self.assertEqual({'tests': False, 'package': False}, ci_scope.classify(['docs/runtime-options.md', 'tests/e5/README.md']))

    def test_packaged_prose_also_checks_package(self):
        for path in ('README.md', 'CHANGELOG.md'):
            self.assertEqual({'tests': False, 'package': True}, ci_scope.classify([path]))

    def test_code_workflow_assets_and_unknown_files_run_all_jobs(self):
        for path in ('src/Lokad.Onnx/Tensor.cs', '.github/workflows/ci.yml', 'icon.png', 'LICENSE.txt',
                     'global.json', 'eng/ci_scope.py', 'tests/opfuzz/reference.json', 'new-file'):
            self.assertEqual({'tests': True, 'package': True}, ci_scope.classify([path]))

    def test_empty_changes_need_no_build(self):
        self.assertEqual({'tests': False, 'package': False}, ci_scope.classify([]))

    def test_initial_push_is_conservative(self):
        self.assertEqual({'tests': True, 'package': True}, ci_scope.classify(ci_scope.changed_paths('0' * 40, 'HEAD')))

    @patch('ci_scope.subprocess.check_output', return_value=b'deleted.cs\0old-name.cs\0docs/new.md\0')
    def test_deletions_and_both_sides_of_renames_count(self, command):
        self.assertTrue(ci_scope.classify(ci_scope.changed_paths('base', 'head'))['tests'])
        self.assertIn('--no-renames', command.call_args.args[0])


if __name__ == '__main__':
    unittest.main()
