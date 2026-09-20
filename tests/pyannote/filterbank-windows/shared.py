"""Reuse the frozen reference arithmetic without editing its original campaign."""
from pathlib import Path
import sys

REFTOOLS = Path(__file__).resolve().parents[1] / 'filterbank-reference'
sys.path.insert(0, str(REFTOOLS))
from common import *
from routes import scalar_window, direct_tables, direct_fourier
from audit import resource_checks

PRIOR_REFERENCE = ROOT / 'artifacts/wespeaker-full-reference-20260920'
REFERENCE_RECEIPT = '25d45f0f2cc3c70b442fb0b1dd6ab4be4420dcf1ce0667a4c232abf83f981ce8'
VARIANTS = ('native', 'windows-default', 'windows-preferred', 'amd-default', 'amd-preferred')
CORPORA = (
    dict(name='five', directory='artifacts/pyannote-diarization-20260919', frozen='frozen', windows=8,
         case_windows=[1, 1, 2, 4, 1], failures=dict(windows=10, amd=10)),
    dict(name='dialogue', directory='artifacts/pyannote-dialogue-20260919', frozen='archive/prepared', windows=24,
         case_windows=[21, 1, 1, 1], failures=dict(windows=19, amd=24)))

def extract(samples, index):
    assert samples.dtype == np.float32 and samples.ndim == 1 and 1 <= samples.size <= 9600000
    assert np.isfinite(samples).all() and np.max(np.abs(samples)) <= 1
    count = 1 if samples.size < 160000 else 1 + (samples.size - 160000 + 15999) // 16000
    assert isinstance(index, int) and 0 <= index < count
    start = index * 16000; valid = min(160000, samples.size - start)
    result = np.zeros(160000, np.float32); result[:valid] = samples[start:start + valid]
    return result, valid

def load_baseline(desc):
    path = ROOT / desc['file']; assert pin(path) == desc['pin']
    value = np.load(path, allow_pickle=False) if desc['format'] == 'npy' else np.fromfile(path, dtype='<f4').reshape(desc['shape'])
    assert value.dtype == np.float32 and list(value.shape) == desc['shape'] and np.isfinite(value).all()
    return value

def feature_inventory(manifest, trace):
    expected = [(case['name'], window['features']) for case in manifest['cases'] for window in case['windows'] if 'features' in window]
    observed = [(row['name'], row['reference']) for row in trace['reports'] if row['stage'] == 'features']
    assert observed == expected and len(set(observed)) == len(observed)
    return {key: row for key, row in zip(observed, (r for r in trace['reports'] if r['stage'] == 'features'))}
