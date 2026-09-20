from pathlib import Path
import sys
REFTOOLS = Path(__file__).resolve().parents[1] / 'filterbank-reference'
sys.path.insert(0, str(REFTOOLS))
from common import *

WINDOWS = ROOT / 'artifacts/wespeaker-window-reference-20260920'
WINDOW_RECEIPT = '3e5130b1a76a626d12c7108ffcab0a9b01bcbb8e72de26e50558cd6c57fc7bed'
CAPTURES = dict(five='artifacts/pyannote-diarization-20260919/frozen/bin',
                dialogue='artifacts/pyannote-dialogue-20260919/archive/prepared/bin')
CONTROL_LIMIT = 1e-12

def baseline(desc):
    path = ROOT / desc['file']; assert pin(path) == desc['pin']
    return np.fromfile(path, dtype='<f4').reshape(desc['shape'])
