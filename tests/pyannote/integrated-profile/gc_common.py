"""Offline GC phase; the closed profile artifacts remain immutable."""
from common import *

GC_BASE = ROOT / 'artifacts/pyannote-integrated-gc-20260922'
OLD_GC = ROOT / 'artifacts/pyannote-retained-gc-20260922'
GC_TOOLS = ROOT / 'tests/pyannote/retained-gc'


def parser():
    path = GC_TOOLS / 'analyze_v3.py'
    spec = importlib.util.spec_from_file_location('integrated_gc_parser', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    value.BASE = GC_BASE
    value.INPUT = BASE
    return value
