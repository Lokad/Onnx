"""Compile generated transport scripts without SSH, writes or inference."""
import ast
import copy
import inspect
from pathlib import Path
import unittest
from unittest.mock import patch
import prepare
import run


class TransportTests(unittest.TestCase):
    def test_every_generated_remote_script_compiles(self):
        namespace = dict(vars(run), owners=[dict(pid=1, birth=2.)],
                         spec=dict(archive=dict(bytes=1, sha256='a'*64), stage=dict(bytes=2, sha256='b'*64)))
        expressions = [node.args[0] for node in ast.walk(ast.parse(inspect.getsource(run.stage)))
                       if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'ssh']
        expressions += [node.value for node in ast.walk(ast.parse(inspect.getsource(run.collect)))
                        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'script' for t in node.targets)]
        self.assertEqual(len(expressions), 4)
        for expression in expressions:
            # Evaluate only each source's string-construction expression. No
            # remote script is executed, and no transport function is called.
            script = eval(compile(ast.Expression(expression), '<transport-string>', 'eval'), namespace)
            compile(script, '<remote-script>', 'exec')
            self.assertIn(run.REMOTE, script)

    def test_original_launcher_and_observer_bound_to_new_namespace(self):
        self.assertIs(run.launch, run.transport.launch)
        self.assertIs(run.observe, run.transport.observe)
        self.assertEqual(run.launch.__globals__['BASE'], prepare.BASE)
        self.assertEqual(run.launch.__globals__['REMOTE'], run.REMOTE)
        self.assertEqual(run.transport.previous_closed, prepare.previous_closed)

    def test_exact_root_binding_refuses_drift(self):
        # Schema fixture only: this does not construct a qualification receipt.
        value = dict(passed=True, root_source_verified=True, consumer=dict(passed=True),
            measured={'Lokad.Onnx.dll': dict(sha256=prepare.CORE), 'Lokad.Onnx.Data.dll': dict(sha256=prepare.DATA)},
            inventory=dict(passed=True, core_methods=3283, data_methods=697, public_surface_equal=True,
                           assembly_attributes_equal=True, method_bodies_equal=True, implementation_flags_equal=True),
            built={'Lokad.Onnx.dll': dict(bytes=1, sha256='c'*64), 'Lokad.Onnx.Data.dll': dict(bytes=1, sha256='d'*64)})
        self.assertEqual(prepare.root_binding(value), value['built'])
        for field in ['core_methods', 'implementation_flags_equal', 'public_surface_equal']:
            changed = copy.deepcopy(value)
            changed['inventory'][field] = 3282 if field == 'core_methods' else False
            with self.assertRaises(AssertionError): prepare.root_binding(changed)
        for field in ['passed', 'root_source_verified']:
            changed = copy.deepcopy(value); changed[field] = False
            with self.assertRaises(AssertionError): prepare.root_binding(changed)
        changed = copy.deepcopy(value); changed['measured']['Lokad.Onnx.dll']['sha256'] = '0'*64
        with self.assertRaises(AssertionError): prepare.root_binding(changed)
        changed = copy.deepcopy(value); changed['built'] = {}
        with self.assertRaises(AssertionError): prepare.root_binding(changed)

    def test_missing_qualification_refused_before_preparation(self):
        with patch.object(Path, 'exists', return_value=False):
            with self.assertRaisesRegex(AssertionError, 'Finish actual root/package qualification first'):
                prepare.previous_closed()


if __name__ == '__main__': unittest.main()
