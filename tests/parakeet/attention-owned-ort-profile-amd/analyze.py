"""Run the original complete native attribution and retain its release context."""
import runpy
from run import BASE, ORIGINAL, TOOLS, pin, read, write, qualification


if __name__ == '__main__':
    context = qualification()
    assert read(BASE/'qualification.json') == context
    assert not (BASE/'context-closed.json').exists()
    runpy.run_path(str(ORIGINAL/'analyze.py'), run_name='__main__')
    assert read(BASE/'closed.json')['passed']
    write(BASE/'context-closed.json', dict(passed=True, attribution_only=True,
        original_attribution=pin(BASE/'closed.json'), qualification=pin(BASE/'qualification.json'),
        adapter=pin(TOOLS/'run.py'), auditor=pin(TOOLS/'analyze.py')))
