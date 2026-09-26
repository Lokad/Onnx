"""Check the staged source directory required by the actual unchanged worker."""
import ast
import unittest
from pathlib import Path
from prepare import GRAPH,PARENT,ROOT,TOOLS,read


class Layout(unittest.TestCase):
    def test_actual_worker_working_directory_has_pinned_input(self):
        worker=ast.parse((PARENT/'remote.py').read_text())
        calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='Popen']
        self.assertEqual(len(calls),1)
        cwd,=[k.value for k in calls[0].keywords if k.arg=='cwd']
        directory=eval(compile(ast.Expression(cwd),'worker-cwd','eval'),{'BASE':Path('bundle')})
        self.assertEqual(directory,Path('bundle/source'))
        original=read(GRAPH/'payload.json');self.assertIn('source/global.json',original['files'])
        from protocol import pin
        self.assertEqual(pin(GRAPH/'collected/source/global.json'),original['files']['source/global.json'])
        preparation=(TOOLS/'prepare.py').read_text()
        condition=next(n.test for n in ast.walk(ast.parse(preparation)) if isinstance(n,ast.If) and 'source/global.json' in ast.unparse(n.test))
        self.assertTrue(eval(compile(ast.Expression(condition),'link-selection','eval'),{'name':'source/global.json'}))
        self.assertIn("assert (BASE/'source/global.json').is_file()",(TOOLS/'remote_prepare.py').read_text())

    def test_products_workers_and_scorers_are_reused(self):
        from consumer_scope import verify_scope
        self.assertTrue(verify_scope())
        self.assertFalse((TOOLS/'remote.py').exists())
        self.assertFalse((TOOLS/'checks.py').exists())
        self.assertFalse((TOOLS/'statistics.py').exists())


if __name__=='__main__':unittest.main()
