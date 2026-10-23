"""Independent SciPy arrays for short clips and coprime sample rates."""
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
    output = parser.parse_args().output
    assert not output.exists(), 'Reference output must be new'
    cases = []
    for source, target, length in ((48000, 16000, 1), (48001, 16000, 1), (191999, 16000, 1),
                                   (16001, 16000, 37), (16000, 16001, 37), (8001, 192000, 1),
                                   (48000, 16000, 257), (16000, 16000, 7)):
        index = np.arange(length, dtype=np.float64)
        samples = (0.4 * np.sin(index * 0.417) + 0.2 * np.cos(index * 1.732)).astype(np.float32)
        samples[0] = 0.5
        factor = math.gcd(source, target)
        up, down = target // factor, source // factor
        half = 32 * max(up, down)
        taps = firwin(2 * half + 1, 0.94 / max(up, down), window=('kaiser', 8.6))
        result = resample_poly(samples.astype(np.float64), up, down, window=taps).astype(np.float32)
        cases.append(dict(source_rate=source, destination_rate=target, input=samples.tolist(), output=result.tolist(),
                          filter_sha256=hashlib.sha256(taps.tobytes()).hexdigest()))
    record = dict(numpy=np.__version__, scipy=scipy.__version__,
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), cases=cases)
    output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print('Saved', len(cases), 'independent short-request reference arrays')


if __name__ == '__main__':
    main()
