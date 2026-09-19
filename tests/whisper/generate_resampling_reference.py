"""Generate independent small full-array references for managed audio conversion."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import scipy
from scipy.signal import firwin, resample_poly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Reference output must be new'
    cases = []
    for source, target, length in ((8000, 16000, 17), (44100, 16000, 149), (48000, 16000, 257),
                                   (16000, 48000, 73), (96000, 22050, 127), (8011, 16000, 31),
                                   (16000, 16000, 39)):
        index = np.arange(length, dtype=np.float64)
        samples = (0.4 * np.sin(index * 0.417) + 0.2 * np.cos(index * 1.732)).astype(np.float32)
        samples[0] = 0.75
        samples[-1] = -0.5
        factor = math.gcd(source, target)
        up, down = target // factor, source // factor
        half = 32 * max(up, down)
        taps = firwin(2 * half + 1, 0.94 / max(up, down), window=('kaiser', 8.6))
        result = resample_poly(samples.astype(np.float64), up, down, window=taps).astype(np.float32)
        cases.append(dict(source_rate=source, destination_rate=target, input=samples.tolist(), output=result.tolist(),
                          filter_sha256=hashlib.sha256(taps.tobytes()).hexdigest()))
    record = dict(numpy=np.__version__, scipy=scipy.__version__, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  contract='centered Kaiser beta8.6; half32*max(up,down); cutoff0.94/max; normalized DC; zero extension', cases=cases)
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Saved', len(cases), 'full-array cases to', args.output)


if __name__ == '__main__':
    main()
