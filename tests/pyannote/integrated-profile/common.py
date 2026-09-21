"""Exact integrated product with the original complete diagnostic protocol."""
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[3]
TOOLS = Path(__file__).resolve().parent
BASE = ROOT / 'artifacts/pyannote-integrated-profile-20260922'
PRIOR = ROOT / 'artifacts/pyannote-sampled-thread-time-20260921'
QUALIFIED = ROOT / 'artifacts/pyannote-portable-applications-20260922'
OLD_TOOLS = ROOT / 'tests/pyannote/sampled-thread-time'
CORE = 'e9c87932b2184c2f6bfef72faabb1719bdbceadc779a15fe1ffd3f3056d02838'
DATA = '85d166b59e2beef18ca7664f76faf445bf3cd81509f8f1d1c4b3c5354f53757a'
OLD_CORE = '0d098ba5fd3fd8799bb1dd018148123f296802c769db5f6148a1a5fa9d80118e'
OLD_DATA = '1d34666456a5da749dc3b40ee25621af0bed806c9167f3ad12e96d0736dca662'
spec = importlib.util.spec_from_file_location('integrated_profile_monitor', OLD_TOOLS / 'common.py')
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)
driver.BASE = BASE
driver.monitor.BASE = BASE
monitor = driver.monitor
pin, read, save, verify, terminal, psutil = driver.pin, driver.read, driver.save, driver.verify, driver.terminal, driver.psutil
pair, new_state, verify_spec = driver.pair, driver.new_state, driver.verify_spec
INPUT, FEED = driver.INPUT, driver.FEED
sys.path.append(str(OLD_TOOLS))


def rel(path):
    return path.relative_to(ROOT).as_posix()


def original(name):
    path = OLD_TOOLS / name
    namespace = dict(globals(), __name__='original_integrated_' + name, __file__=str(path))
    exec(compile(path.read_text(encoding='utf8'), str(path), 'exec'), namespace)
    namespace['main']()
