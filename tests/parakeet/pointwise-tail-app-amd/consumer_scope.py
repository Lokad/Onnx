"""Bind original scoring and model checks; adapt only evidence and paths."""
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
PARENT = TOOLS.parent/'decoder-lstm-layout-app-amd'
TRANSPORT = TOOLS.parent/'owned-batch-isolation-parent-app-amd'
NAMES = ['protocol.py','remote.py','statistics_exact.py','audit.py','checks.py','test_admission.py']


def prerequisite_tail():
    marker = "    assert models['consumers']['AudioBenchmark']"
    text = (PARENT/'prerequisites.py').read_text()
    assert text.count(marker) == 1
    tail = marker+text.split(marker,1)[1]
    original = (TOOLS.parent/'decoder-packed-row-app-amd/prerequisites.py').read_text()
    assert tail == marker+original.split(marker,1)[1]
    return tail


def remote_preparation():
    text = (PARENT/'remote_prepare.py').read_text()
    for before,after in [
        ('/dev/shm/lokad-lstmlayout-models-20260927','/dev/shm/lokad-pwt-models-20260927'),
        ('LSTM layout candidate','pointwise remainder candidate')]:
        assert text.count(before) == (2 if before == 'LSTM layout candidate' else 1)
        text = text.replace(before,after)
    assert text.count('copy_function=os.link') == 2
    text = text.replace('copy_function=os.link','copy_function=link_retained')
    marker = '\ndef main():\n'
    assert text.count(marker) == 1
    text = text.replace(marker, '''
def link_retained(source, destination):
    source = Path(source).resolve(); destination = Path(destination)
    if source.stat().st_dev == destination.parent.stat().st_dev: os.link(source, destination)
    else: shutil.copy2(source, destination)
    assert pin(source) == pin(destination)
    return str(destination)

def main():
''')
    return text


def verify_scope():
    original = TOOLS.parent/'decoder-packed-row-app-amd'
    for name in NAMES: assert (PARENT/name).read_bytes() == (original/name).read_bytes(),name
    marker = "    assert models['consumers']['AudioBenchmark']"
    assert marker+(TOOLS/'prerequisites.py').read_text().split(marker,1)[1] == prerequisite_tail()
    remote_preparation()
    return True
